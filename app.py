from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
import os
import json
import hashlib
import re
import datetime
import logging
import time
import threading
import random

logger = logging.getLogger("midan")
from typing import List, Optional, Dict

from fastapi import FastAPI, Request, Form, Depends, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from sqlalchemy import text as sql_text, func
from openai import OpenAI

from database import (
    SessionLocal,
    init_db,
    Category,
    Question,
    MediaCandidate,
    AuditLog,
    LearningEvent,
)

from agents.media_agent import run_media_agent_for_question_ids, build_media_hint
from agents.factcheck_agent import run_factcheck_agent_for_question_ids
from agents.validate_agent import run_validate_agent_for_question_ids

# -------------------------------------------------------------------
# Basic config
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MEDIA_DIR = os.path.join(STATIC_DIR, "media")
ORIGINAL_MEDIA_DIR = os.path.join(MEDIA_DIR, "original")

# Optional persistent media directory (Render disk, etc.)
# If set, AI-generated images are stored there and served via /persist-media.
PERSIST_MEDIA_DIR = (os.environ.get("PERSIST_MEDIA_DIR") or "").strip()
if PERSIST_MEDIA_DIR:
    try:
        os.makedirs(PERSIST_MEDIA_DIR, exist_ok=True)
    except Exception:
        PERSIST_MEDIA_DIR = ""

os.makedirs(ORIGINAL_MEDIA_DIR, exist_ok=True)

app = FastAPI()

# Serialize SQLite write transactions (single-writer DB)
DB_WRITE_LOCK = threading.RLock()


app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET_KEY", "change-me-please"),
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if PERSIST_MEDIA_DIR:
    app.mount("/persist-media", StaticFiles(directory=PERSIST_MEDIA_DIR), name="persist-media")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY not set – AI features will fail.")
client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------------------------------------------------------
# DB session dependency
# -------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------------------------------------------------
# Simple helpers
# -------------------------------------------------------------------


def _db_retry(fn, *, retries: int = 8, base_sleep: float = 0.08, label: str = "db"):
    """Retry wrapper for transient SQLite 'database is locked' errors."""
    last_exc = None
    for i in range(max(1, retries)):
        try:
            with DB_WRITE_LOCK:
                return fn()
        except OperationalError as e:
            last_exc = e
            msg = str(e).lower()
            if "database is locked" in msg or "locked" in msg:
                # Exponential backoff + jitter
                sleep_s = base_sleep * (2 ** i) + (random.random() * 0.05)
                logger.warning(f"[{label}] sqlite locked; retry {i+1}/{retries} in {sleep_s:.2f}s")
                time.sleep(min(sleep_s, 2.0))
                continue
            raise
    if last_exc:
        raise last_exc



def _ensure_media_reference(stem_en: str, stem_ar: str, media_type: str) -> bool:
    """Media-first constraint: question stem must explicitly reference the media."""
    mt = (media_type or "").lower().strip()
    en = (stem_en or "").lower()
    ar = (stem_ar or "")
    en_tokens = ["image", "photo", "picture", "logo", "audio", "sound", "clip", "video"]
    ar_tokens = ["صورة", "الصورة", "هذه الصورة", "الشعار", "شعار", "فيديو", "مقطع", "صوت", "تسجيل", "اللقطة"]
    # strengthen based on type
    if mt == "logo":
        en_tokens += ["logo"]
        ar_tokens += ["الشعار", "شعار"]
    if mt == "audio":
        en_tokens += ["audio", "sound"]
        ar_tokens += ["صوت", "تسجيل"]
    if mt in ("video", "youtube"):
        en_tokens += ["video", "clip"]
        ar_tokens += ["فيديو", "مقطع"]
    ok = any(t in en for t in en_tokens) or any(t in ar for t in ar_tokens)
    return ok

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _normalize_question_text(stem_en: str, stem_ar: str, answer_en: str, answer_ar: str) -> str:
    parts = [
        (stem_en or "").strip().lower(),
        (stem_ar or "").strip().lower(),
        (answer_en or "").strip().lower(),
        (answer_ar or "").strip().lower(),
    ]
    return " | ".join(p for p in parts if p)


def _normalize_stem(stem_en: str, stem_ar: str) -> str:
    """Normalize stem for de-duplication. Arabic is primary; EN is optional."""
    s = " ".join([(stem_en or ""), (stem_ar or "")]).strip().lower()
    s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _stem_hash(stem_en: str, stem_ar: str) -> str:
    return hashlib.sha1(_normalize_stem(stem_en, stem_ar).encode("utf-8")).hexdigest()


def _current_user(request: Request) -> Optional[str]:
    return request.session.get("user")


get_current_user = _current_user

def _require_login(request: Request) -> None:
    if not _current_user(request):
        raise HTTPException(status_code=401, detail="Not authenticated")


def _log_audit(
    db: Session,
    user: Optional[str],
    action: str,
    entity: str,
    entity_id: Optional[int],
    after_details: Optional[Dict] = None,
    before_details: Optional[Dict] = None,
    message: Optional[str] = None,
):
    """Write an audit record.

    - action: machine-friendly action (e.g., ai_generate, media_select, question_edit)
    - message: optional human-friendly message; defaults to action
    """
    try:
        entry = AuditLog(
            actor=user or "system",
            action=(action or "").strip() or None,
            message=(message or action or "").strip() or "event",
            entity=entity,
            entity_id=entity_id,
            before_json=(json.dumps(before_details, ensure_ascii=False) if before_details is not None else None),
            after_json=(json.dumps(after_details, ensure_ascii=False) if after_details is not None else None),
            created_at=datetime.datetime.datetime.utcnow(),
        )
        db.add(entry)
        db.commit()
    except Exception:
        # Audit logging must never break core product flows
        db.rollback()


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    init_db()
    print("[INIT] Database initialized.")


# -------------------------------------------------------------------
# Authentication (very simple)
# -------------------------------------------------------------------

# ----------------------------
# LOGIN (Arabic-first UI)
# ----------------------------

from fastapi import Form, Request
from fastapi.responses import RedirectResponse
from starlette.status import HTTP_302_FOUND
import os

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS") or os.getenv("QB_ADMIN_PASSWORD") or "admin"  # default for local/demo


@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )


@app.post("/login")
async def login(
    request: Request,
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None)
):
    """
    Accepts credentials from either:
      1) HTML form POST (application/x-www-form-urlencoded or multipart/form-data), or
      2) JSON body (application/json) for API/JS clients.
    """

    # If the request is JSON (or the form fields were not provided), try JSON parsing.
    if username is None or password is None:
        try:
            data = await request.json()
            if isinstance(data, dict):
                username = data.get("username")
                password = data.get("password")
        except Exception:
            pass

    # Basic validation (avoid FastAPI "Field required" errors for empty bodies)

    # If the UI only posts a password (common in simple admin panels),
    # treat the username as "admin" by default.
    if (not username) and password:
        username = "admin"

    if not username or not password:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Username and password are required."},
        )

    if username == ADMIN_USER and password == ADMIN_PASS:
        request.session["user"] = username
        return RedirectResponse("/", status_code=HTTP_302_FOUND)

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid username or password"},
    )





@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# -------------------------------------------------------------------
# Dashboard with Audit
# -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    if not _current_user(request):
        return RedirectResponse("/login", status_code=303)

    total_categories = db.query(Category).count()
    total_questions = db.query(Question).count()

        # Live media counts (source of truth: MediaCandidate table)
    # Pending: MEDIA questions with no candidates at all
    # Review: candidates exist but none selected
    # Approved: at least one selected candidate
    # Live media counts
    mc_any = db.query(MediaCandidate.question_id).filter(MediaCandidate.question_id == Question.id).exists()
    pending_media = db.query(Question).filter(Question.question_mode == "MEDIA").filter(~mc_any).count()
    media_review = db.query(Question).filter(Question.question_mode == "MEDIA").filter(mc_any).filter(Question.media_status != "APPROVED").count()
    media_approved = db.query(Question).filter(Question.question_mode == "MEDIA").filter(Question.media_status == "APPROVED").count()

    active_questions = db.query(Question).filter(Question.status == "active").count()

    pending_factcheck = db.query(Question).filter(Question.status == "needs_factcheck").count()
    pending_validation = db.query(Question).filter(Question.status == "needs_validation").count()

    # Build per-category stats for dashboard table
    categories = db.query(Category).order_by(Category.id.asc()).all()
    cat_stats = []
    for c in categories:
        total = db.query(Question).filter(Question.category_id == c.id).count()
        active = db.query(Question).filter(Question.category_id == c.id, Question.status == "active").count()
        draft = db.query(Question).filter(Question.category_id == c.id, Question.status != "active").count()

                # Live media counts per category (MediaCandidate is source of truth)
        c_q = db.query(Question).filter(Question.category_id == c.id, Question.question_mode == "MEDIA")
        c_pending = c_q.filter(~mc_any).count()
        c_review = c_q.filter(mc_any).filter(Question.media_status != "APPROVED").count()
        c_approved = c_q.filter(Question.media_status == "APPROVED").count()
        c_media_approved = c_approved
        c_media_pending = c_pending
        c_media_review = c_review

        cat_stats.append(
            {
                "id": c.id,
                "name_en": c.name_en,
                "name_ar": c.name_ar,
                "subtopic": c.subtopic,
"total": total,
                "active": active,
                "draft": draft,
                "media_approved": c_media_approved,
                "media_pending": c_media_pending,
                "media_review": c_media_review,
            }
        )

    # Recent audit
    recent_audit = (
        db.query(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(10)
        .all()
    )

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": _current_user(request),
            "total_categories": total_categories,
            "total_questions": total_questions,
            "active_questions": active_questions,
            "pending_media": int(pending_media or 0),
            "media_review": media_review,
            "review_media": int(media_review or 0),
            "media_approved": media_approved,
            "approved_media": int(media_approved or 0),
            "pending_factcheck": pending_factcheck,
            "pending_validation": pending_validation,
"cat_stats": cat_stats,
            "recent_audit": recent_audit,
            "logs": recent_audit,
        },
    )

@app.get("/audit", response_class=HTMLResponse)
def audit_log(request: Request, db: Session = Depends(get_db)):
    _require_login(request)
    logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(200).all()
    return templates.TemplateResponse(
        "audit.html",
        {"request": request, "user": _current_user(request), "logs": logs},
    )

@app.get("/categories", response_class=HTMLResponse)
def categories_list(request: Request, db: Session = Depends(get_db)):
    _require_login(request)
    cats = db.query(Category).order_by(Category.id.desc()).all()
    return templates.TemplateResponse(
        "categories.html",
        {"request": request, "categories": cats, "user": _current_user(request)},
    )


@app.get("/categories/new", response_class=HTMLResponse)
def categories_new_form(request: Request):
    _require_login(request)
    return templates.TemplateResponse(
        "category_form.html",
        {"request": request, "category": None, "user": _current_user(request)},
    )


@app.post("/categories/new")
def categories_new_post(
    request: Request,
    name_en: str = Form(""),
    name_ar: str = Form(""),
    subtopic: str = Form(""),
    description_en: str = Form(""),
    description_ar: str = Form(""),
    db: Session = Depends(get_db),
):
    _require_login(request)

    name_en = (name_en or "").strip()
    name_ar = (name_ar or "").strip()

    if not name_en and not name_ar:
        raise HTTPException(status_code=400, detail="Please provide Category name in Arabic and/or English.")

    # Deduplicate by EN (case-insensitive) when present; otherwise by AR exact match.
    q = db.query(Category)
    existing = None
    if name_en:
        existing = q.filter(Category.name_en.ilike(name_en)).first()
    elif name_ar:
        existing = q.filter(Category.name_ar == name_ar).first()

    if existing:
        return RedirectResponse("/categories", status_code=303)

    cat = Category(
        name_en=name_en or None,
        name_ar=name_ar or None,
        subtopic=(subtopic or "").strip() or None,
        description_en=(description_en or "").strip() or None,
        description_ar=(description_ar or "").strip() or None,
        created_at=datetime.datetime.datetime.utcnow(),
    )
    db.add(cat)
    db.commit()

    _log_audit(
        db,
        _current_user(request),
        "create",
        "Category",
        cat.id,
        {
            "name_en": cat.name_en,
            "name_ar": cat.name_ar,
            "subtopic": cat.subtopic,
            "description_en": cat.description_en,
            "description_ar": cat.description_ar,
        },
        None,
    )

    return RedirectResponse("/categories", status_code=303)

@app.get("/categories/{category_id}/edit", response_class=HTMLResponse)
def categories_edit_form(
    request: Request,
    category_id: int,
    db: Session = Depends(get_db),
):
    _require_login(request)
    cat = db.query(Category).get(category_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    def _is_recipe_category(cat_obj: Category) -> bool:
        t = " ".join([(cat_obj.name_en or ""), (cat_obj.name_ar or "")]).lower()
        kws = ["recipe", "recipes", "cooking", "cook", "cuisine", "dish", "food", "dessert", "baking", "kitchen",
               "وصفة", "وصفات", "طبخ", "مطبخ", "اكل", "طعام", "حلوى"]
        return any(k in t for k in kws)

    is_recipe_category = _is_recipe_category(cat)

    return templates.TemplateResponse(
        "category_form.html",
        {"request": request, "category": cat, "user": _current_user(request)},
    )


@app.post("/categories/{category_id}/edit")
def categories_edit_post(
    request: Request,
    category_id: int,
    name_en: str = Form(""),
    name_ar: str = Form(""),
    subtopic: str = Form(""),
    description_en: str = Form(""),
    description_ar: str = Form(""),
    db: Session = Depends(get_db),
):
    _require_login(request)

    cat = db.query(Category).get(category_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    def _is_recipe_category(cat_obj: Category) -> bool:
        t = " ".join([(cat_obj.name_en or ""), (cat_obj.name_ar or "")]).lower()
        kws = ["recipe", "recipes", "cooking", "cook", "cuisine", "dish", "food", "dessert", "baking", "kitchen",
               "وصفة", "وصفات", "طبخ", "مطبخ", "اكل", "طعام", "حلوى"]
        return any(k in t for k in kws)

    is_recipe_category = _is_recipe_category(cat)

    name_en = (name_en or "").strip()
    name_ar = (name_ar or "").strip()
    if not name_en and not name_ar:
        raise HTTPException(status_code=400, detail="Please provide Category name in Arabic and/or English.")

    old = {
        "name_en": cat.name_en,
        "name_ar": cat.name_ar,
        "subtopic": cat.subtopic,
        "description_en": cat.description_en,
        "description_ar": cat.description_ar,
    }

    cat.name_en = name_en or None
    cat.name_ar = name_ar or None
    cat.subtopic = (subtopic or "").strip() or None
    cat.description_en = (description_en or "").strip() or None
    cat.description_ar = (description_ar or "").strip() or None

    db.commit()

    _log_audit(
        db,
        _current_user(request),
        "update",
        "Category",
        cat.id,
        old,
        {
            "name_en": cat.name_en,
            "name_ar": cat.name_ar,
            "subtopic": cat.subtopic,
            "description_en": cat.description_en,
            "description_ar": cat.description_ar,
        },
    )

    return RedirectResponse("/categories", status_code=303)

@app.get("/categories/{category_id}/subtopics")
def category_subtopics(category_id: int, db: Session = Depends(get_db)):
    """Return a list of subtopic suggestions for a category.

    Uses:
    - Category.subtopic seed (if present)
    - Distinct Question.subtopic values previously used for the category
    """
    cat = db.query(Category).filter(Category.id == category_id).first()
    if not cat:
        return {"subtopics": []}

    subs = set()
    if cat.subtopic:
        subs.add(cat.subtopic.strip())

    rows = (
        db.query(Question.subtopic)
        .filter(Question.category_id == category_id)
        .filter(Question.subtopic.isnot(None))
        .all()
    )
    for (s,) in rows:
        if s and str(s).strip():
            subs.add(str(s).strip())

    return {"subtopics": sorted(subs)}

@app.get("/categories/{category_id}/questions", response_class=HTMLResponse)
def category_questions(
    request: Request,
    category_id: int,
    status: Optional[str] = None,
    q: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Alias route used by some templates/links."""
    _require_login(request)
    return questions_list(request=request, category_id=category_id, status=status, q=q, db=db)


@app.get("/questions", response_class=HTMLResponse)
def questions_list(
    request: Request,
    category_id: Optional[int] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
    db: Session = Depends(get_db),
):
    _require_login(request)

    query = db.query(Question).order_by(Question.id.desc())

    if category_id:
        query = query.filter(Question.category_id == category_id)
    if status:
        query = query.filter(Question.status == status)
    if q:
        like = f"%{q}%"
        query = query.filter(
            (Question.stem_en.ilike(like))
            | (Question.stem_ar.ilike(like))
            | (Question.answer_en.ilike(like))
            | (Question.answer_ar.ilike(like))
        )

    questions = query.limit(500).all()
    cats = db.query(Category).all()
    cats_map = {c.id: c for c in cats}

    return templates.TemplateResponse(
        "questions.html",
        {
            "request": request,
            "questions": questions,
            "categories": cats,
            "cats_map": cats_map,
            "filter_category_id": category_id,
            "filter_status": status,
            "filter_q": q,
            "user": _current_user(request),
        },
    )


@app.get("/questions/all", response_class=HTMLResponse)
def questions_all(
    request: Request,
    db: Session = Depends(get_db),
):
    _require_login(request)
    questions = db.query(Question).order_by(Question.id.desc()).limit(2000).all()
    cats = db.query(Category).all()
    cats_map = {c.id: c for c in cats}
    return templates.TemplateResponse(
        "questions_all.html",
        {
            "request": request,
            "questions": questions,
            "categories": cats,
            "cats_map": cats_map,
            "user": _current_user(request),
        },
    )


@app.get("/questions/{question_id}/edit", response_class=HTMLResponse)
def question_edit_form(
    request: Request,
    question_id: int,
    db: Session = Depends(get_db),
):
    _require_login(request)

    q = db.query(Question).get(question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    cats = db.query(Category).all()
    cat = next((c for c in cats if c.id == q.category_id), None)

    hint_text = build_media_hint(
        cat.name_en if cat else "",
        q.subtopic or (cat.subtopic if cat else "") or "",
        q.hint or "",
        q.stem_en or "",
        q.answer_en or "",
        q.question_type or "",
        q.region or "saudi",
    )

    media_candidates = (
        db.query(MediaCandidate)
        .filter(MediaCandidate.question_id == q.id)
        .order_by(MediaCandidate.score.desc())
        .all()
    )

    return templates.TemplateResponse(
        "question_edit.html",
        {
            "request": request,
            "question": q,
            "categories": cats,
            "hint_text": hint_text,
            "media_candidates": media_candidates,
            "candidates": media_candidates,
            "user": _current_user(request),
        },
    )


@app.post("/questions/{question_id}/edit")
def question_edit_post(
    request: Request,
    question_id: int,
    category_id: int = Form(...),
    stem_en: str = Form(""),
    stem_ar: str = Form(""),
    answer_en: str = Form(""),
    answer_ar: str = Form(""),
    option1_en: str = Form(""),
    option2_en: str = Form(""),
    option3_en: str = Form(""),
    option4_en: str = Form(""),
    option1_ar: str = Form(""),
    option2_ar: str = Form(""),
    option3_ar: str = Form(""),
    option4_ar: str = Form(""),
    correct_option_index: Optional[int] = Form(None),
    difficulty: str = Form("medium"),
    # NOTE: form field name is "question_type" (template). If we use a different param name,
    # FastAPI will not bind and will fall back to the default, causing type to reset to "text".
    question_type: str = Form("text"),
    game_type: str = Form(""),
    answer_type: str = Form("text"),  # legacy
    region: str = Form("saudi"),
    saudi_ratio: float = Form(1.0),
    current_affairs: bool = Form(False),
    subtopic: str = Form(""),
    hint: str = Form(""),
    status: str = Form("draft"),
    media_query: str = Form(""),
    media_url: str = Form(""),
    media_type: str = Form(""),
    media_status: str = Form("none"),
    media_selected_source: str = Form(""),
    media_selected_score: str = Form(""),
    selected_candidate_id: int = Form(None),
    # Template uses name="media_file" for uploads.
    media_file: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    _require_login(request)

    q = db.query(Question).get(question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    old = {
        "stem_en": q.stem_en,
        "stem_ar": q.stem_ar,
        "answer_en": q.answer_en,
        "answer_ar": q.answer_ar,
        "status": q.status,
        "media_url": getattr(q, "media_url", None) or getattr(q, "url", None),
    }

    q.category_id = category_id
    q.stem_en = (stem_en or "").strip()
    q.stem_ar = (stem_ar or "").strip()

    # Media-first constraint: for MEDIA questions, stem must reference the media
    try:
        if (getattr(q, "question_mode", "") or "").upper() == "MEDIA":
            mt = (getattr(q, "media_type", "") or q.question_type or "").lower().strip()
            if not _ensure_media_reference(q.stem_en, q.stem_ar, mt):
                if q.stem_en:
                    q.stem_en = f"In this {mt or 'media'}, {q.stem_en}".strip()
                if q.stem_ar:
                    q.stem_ar = f"في هذه {'الصورة' if mt in ('picture','logo') else 'الوسائط'}, {q.stem_ar}".strip()
    except Exception:
        pass
    q.answer_en = (answer_en or "").strip()
    q.answer_ar = (answer_ar or "").strip()

    q.option1_en = (option1_en or "").strip()
    q.option2_en = (option2_en or "").strip()
    q.option3_en = (option3_en or "").strip()
    q.option4_en = (option4_en or "").strip()

    q.option1_ar = (option1_ar or "").strip()
    q.option2_ar = (option2_ar or "").strip()
    q.option3_ar = (option3_ar or "").strip()
    q.option4_ar = (option4_ar or "").strip()

    q.correct_option_index = correct_option_index
    q.difficulty = (difficulty or "medium").strip()
    q.question_type = (question_type or "text").strip()
    # Game Type (preferred) + legacy Answer Type
    chosen_game_type = (game_type or answer_type or "mcq_selection").strip()
    if hasattr(q, "game_type"):
        q.game_type = chosen_game_type

    # IMPORTANT: "be_loud" is a *game label* but the input is still text-based.
    legacy_answer_type = "text_input" if chosen_game_type == "be_loud" else chosen_game_type

    # Keep legacy answer_type for older exports/UI (and for UI that still reads answer_type)
    q.answer_type = legacy_answer_type
    q.region = (region or "saudi").strip()
    q.saudi_ratio = float(saudi_ratio) if saudi_ratio is not None else 1.0
    q.current_affairs = bool(current_affairs)
    q.subtopic = (subtopic or "").strip()
    q.hint = (hint or "").strip()
    q.status = (status or "draft").strip()
    q.media_query = (media_query or "").strip()

    entered_media_url = (media_url or "").strip()
    if entered_media_url:
        q.url = entered_media_url

    # If user chose an AI candidate, prefer that URL and mark it selected.
    if selected_candidate_id:
        cand = db.query(MediaCandidate).get(int(selected_candidate_id))
        if cand and cand.question_id == q.id and cand.url:
            db.query(MediaCandidate).filter(MediaCandidate.question_id == q.id).update({"selected": False})
            cand.selected = True

            q.url = cand.url
            q.media_source = cand.source
            q.media_selected_source = cand.source
            q.media_selected_score = str(cand.score) if cand.score is not None else ""
            q.media_status = "PENDING"  # never auto-approve in v15

    # If user uploaded media, store it locally and override URL
    if media_file is not None and getattr(media_file, "filename", ""):
        uploads_dir = Path("static") / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", media_file.filename)
        fname = f"q{q.id}_{safe_name}"
        dest = uploads_dir / fname
        with dest.open("wb") as f:
            f.write(media_file.file.read())
        q.url = f"/static/uploads/{fname}"
        q.media_source = "upload"
        q.media_selected_source = "upload"
        q.media_selected_score = ""
        q.media_status = "PENDING"

    # Backward compatibility for older schemas that still have media_url
    if hasattr(q, "media_url"):
        q.media_url = q.url
    q.media_type = (media_type or "").strip()
    ms = (media_status or "").strip()
    if ms and ms.lower() != "none":
        q.media_status = ms
    # else keep existing status
    mss = (media_selected_source or "").strip()
    if mss:
        q.media_selected_source = mss
    msc = (media_selected_score or "").strip()
    if msc:
        q.media_selected_score = msc

    norm = _normalize_question_text(q.stem_en, q.stem_ar, q.answer_en, q.answer_ar)
    q.dedup_key = _sha(norm) if norm else None

    db.commit()

    _log_audit(
        db,
        _current_user(request),
        "question_edit",
        "Question",
        q.id,
        {"old": old, "new": {
            "stem_en": q.stem_en,
            "stem_ar": q.stem_ar,
            "answer_en": q.answer_en,
            "answer_ar": q.answer_ar,
            "status": q.status,
            "media_url": getattr(q, "media_url", None) or getattr(q, "url", None),
        }},
    )

    # Self-learning: store moderator edits as training signals (for future adaptive generation)
    try:
        from database import LearningEvent
        db.add(
            LearningEvent(
                category_id=q.category_id,
                question_id=q.id,
                actor=_current_user(request),
                event_type="question_edit_save",
                before_json=json.dumps(old, ensure_ascii=False),
                after_json=json.dumps({
                    "stem_en": q.stem_en,
                    "stem_ar": q.stem_ar,
                    "answer_en": q.answer_en,
                    "answer_ar": q.answer_ar,
                    "question_type": q.question_type,
                    "game_type": getattr(q, "game_type", None) or q.answer_type,
                    "media_url": getattr(q, "media_url", None) or getattr(q, "url", None),
                }, ensure_ascii=False),
            )
        )
        db.commit()
    except Exception:
        db.rollback()


    return RedirectResponse(f"/questions?category_id={q.category_id}&saved=1", status_code=303)


# -------------------------------------------------------------------
# AI question generation
# -------------------------------------------------------------------

@app.get("/ai/generate", response_class=HTMLResponse)
def ai_generate_form(request: Request, db: Session = Depends(get_db)):
    _require_login(request)
    cats = db.query(Category).order_by(Category.name_en.asc()).all()
    # Optional pre-select via query param
    selected_category_id = request.query_params.get("category_id")
    message = request.query_params.get("message")
    return templates.TemplateResponse(
        "ai_generate.html",
        {
            "request": request,
            "title": "AI Generate",
            "categories": cats,
            "selected_category_id": selected_category_id,
            "message": message,
        },
    )


@app.post("/ai/generate", response_class=HTMLResponse)
def ai_generate_post(
    request: Request,
    db: Session = Depends(get_db),
    category_id: int = Form(...),
    difficulty: str = Form("medium"),
    region: str = Form("saudi"),  # saudi/global/mixed
    saudi_ratio: float = Form(1.0),  # 0..1
    current_affairs: bool = Form(False),
    question_mode: str = Form("TEXT"),  # TEXT or MEDIA
    media_type: str = Form("picture"),  # used only when MEDIA
    game_type: str = Form("mcq_selection"),  # mcq_selection/text_input/be_loud
    answer_type: str = Form(""),  # legacy (ignored when game_type present)
    count: int = Form(5),
    subtopic: str = Form(""),
    hint: str = Form(""),
):
    """
    Production AI generation.

    Guarantees:
    - Never silently "succeeds" with 0 questions unless count<=0 or category missing.
    - Duplicates are counted as errors (skip reasons exposed via query params).
    - MEDIA mode downgrades to TEXT if insufficient unique media-based questions.
    """
    _require_login(request)

    cat = db.query(Category).get(category_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    # Acquire a write lock early (SQLite is single-writer). This prevents random "database is locked" during flush.
    with DB_WRITE_LOCK:
        try:
            # Reserve the write lock up-front for this request
            db.execute(sql_text("BEGIN IMMEDIATE"))
        except Exception:
            # If a transaction is already open, we just proceed under the Python lock
            pass

    # -----------------------------
    # Normalize inputs
    # -----------------------------
    question_mode = (question_mode or "TEXT").upper().strip()
    if question_mode not in ("TEXT", "MEDIA"):
        question_mode = "TEXT"

    media_type = (media_type or "picture").lower().strip()
    if media_type not in ("logo", "picture", "audio", "video", "youtube"):
        media_type = "picture"

    difficulty = (difficulty or "medium").lower().strip()
    if difficulty not in ("easy", "medium", "hard", "expert", "ladder"):
        difficulty = "medium"

    game_type = (game_type or answer_type or "mcq_selection").lower().strip()
    is_mcq = game_type.startswith("mcq")

    try:
        target_per_diff = int(count or 0)
    except Exception:
        target_per_diff = 0
    if target_per_diff <= 0:
        return RedirectResponse(f"/questions?category_id={category_id}&generated=0&errors=0", status_code=303)

    diffs_to_generate = [difficulty]
    if difficulty == "ladder":
        diffs_to_generate = ["easy", "medium", "hard"]

    # -----------------------------
    # Category helpers
    # -----------------------------
    def _is_recipe_category(cat_obj: Category) -> bool:
        t = " ".join([(cat_obj.name_en or ""), (cat_obj.name_ar or "")]).lower()
        kws = [
            "recipe", "recipes", "cooking", "cook", "cuisine", "dish", "food", "dessert", "baking", "kitchen",
            "وصفة", "وصفات", "طبخ", "مطبخ", "اكل", "طعام", "حلوى",
        ]
        return any(k in t for k in kws)

    is_recipe_category = _is_recipe_category(cat)

    # -----------------------------
    # Prompt blocks
    # -----------------------------
    recipe_rules = (
        "RECIPE RULES (apply only if category is recipes/cooking):\n"
        "- Focus on real cooking/recipes: ingredients, quantities, techniques, tools, cuisines, cooking times/temperatures, food safety.\n"
        "- Avoid vague questions like 'What is a recipe?' or opinion questions ('best', 'tastiest').\n"
        "- If MCQ: distractors must be plausible ingredients/techniques of the same type.\n"
        "- Do not put the correct answer inside the stem.\n"
    )

    common_rules = (
        "STRICT RULES (must comply):\n"
        "1) Questions must be factual and have a single clear correct answer.\n"
        "2) DO NOT generate symbolic, philosophical, interpretive, or opinion-based questions.\n"
        "3) Avoid 'what does it mean/represent/symbolize' style.\n"
        "4) The hint is a HARD constraint; follow it.\n"
        "5) Difficulty must be strictly adhered to.\n"
    )

    difficulty_rules = {
        "easy": "- Easy: direct identification or simple facts. Use obvious cues.\n",
        "medium": "- Medium: factual, avoid extremely easy prompts. Distractors must be close.\n",
        "hard": "- Hard: factual but less obvious; require specific detail recognition. Use close distractors.\n",
        "expert": (
            "- Expert: factual, precise, single-answer; avoid ambiguity or interpretive/philosophical questions.\n"
            "- Avoid extremely obvious household examples.\n"
            "- Target deep but verifiable details.\n"
            "- For MCQ: use 3 close distractors from the same universe/industry/era.\n"
        ),
    }

    topic_line = f"Category (topic): {cat.name_en or cat.name_ar or '(unnamed)'}\n"
    topic_ar_line = f"Category Arabic: {cat.name_ar or cat.name_en or '(unnamed)'}\n"
    region_line = f"Region scope: {region or 'saudi'}\n"
    saudi_ratio_line = f"Saudi ratio (0..1): {saudi_ratio}\n"
    current_affairs_line = f"Current affairs: {'YES' if current_affairs else 'NO'}\n"
    hint_line = f"Hint/Prompt: {(hint or '').strip() or '(none)'}\n"
    subtopic_line = f"Subtopic constraint: {(subtopic or '').strip() or '(none)'}\n"

    extra_rules = recipe_rules if is_recipe_category else ""
    media_intent_extra = (
        "For recipes: media_search_query MUST target ingredients (e.g., 'ingredients flat lay', 'raw ingredients top view'), not only plated dishes.\n"
        if is_recipe_category else ""
    )

    system_msg = (
        "You are a production question generator for a Saudi family-safe trivia game. "
        "You output ONLY valid JSON, no markdown. "
        "You must follow instructions exactly. "
        "Arabic must be Modern Standard Arabic."
    )

    def _parse_json(raw: str):
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if "\n" in raw:
                raw = raw.split("\n", 1)[1]
        return json.loads(raw)

    def _llm(messages, temperature=0.2):
        nonlocal openai_quota_exceeded
        try:
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
                messages=messages,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e)
            # OpenAI quota / billing / 429 insufficient_quota
            if ("insufficient_quota" in msg) or ("exceeded your current quota" in msg) or ("Error code: 429" in msg):
                openai_quota_exceeded = True
                raise RuntimeError("OPENAI_QUOTA_EXCEEDED") from e
            raise


    def _recent_stems_block(limit: int = 30) -> str:
        try:
            q = db.query(Question.stem_ar, Question.stem_en).filter(Question.category_id == cat.id)
            if subtopic and subtopic.strip():
                q = q.filter(Question.subtopic == subtopic.strip())
            rows = q.order_by(Question.id.desc()).limit(limit).all()
            lines = []
            for ar, en in rows:
                t = (en or ar or "").strip()
                if t:
                    lines.append(f"- {t}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _learning_examples_block(limit: int = 5) -> str:
        try:
            rows = (
                db.query(LearningEvent.after_json)
                .filter(LearningEvent.category_id == cat.id, LearningEvent.after_json.isnot(None))
                .order_by(LearningEvent.id.desc())
                .limit(limit)
                .all()
            )
            examples = []
            for (after_json,) in rows:
                try:
                    obj = json.loads(after_json or "{}")
                    stem = (obj.get("stem_en") or obj.get("stem_ar") or "").strip()
                    ans = (obj.get("answer_en") or obj.get("answer_ar") or "").strip()
                    if stem and ans:
                        examples.append(f"Q: {stem}\nA: {ans}")
                except Exception:
                    continue
            return "\n\n".join(examples)
        except Exception:
            return ""

    # -----------------------------
    # Counters / summary
    # -----------------------------
    created = 0
    created_ids: list[int] = []
    errors: list[str] = []
    dup_skips = 0
    media_failures = 0
    parse_failures = 0
    openai_quota_exceeded = False

    recent_block = _recent_stems_block(limit=30)
    learning_block = _learning_examples_block(limit=5)

    # -----------------------------
    # Insert helpers
    # -----------------------------
    def _already_exists(stem_en_val: str, stem_ar_val: str, qtype_key: str, ans_en_val: str, ans_ar_val: str):
        nonlocal dup_skips
        sh = _stem_hash(stem_en_val, stem_ar_val)
        exists = db.query(Question.id).filter(
            Question.category_id == cat.id,
            Question.subtopic == ((subtopic.strip() or None)),
            Question.question_type == qtype_key,
            Question.stem_hash == sh,
        ).first()
        if exists:
            dup_skips += 1
            return True, sh, None

        norm_key = _normalize_question_text(stem_en_val, stem_ar_val, ans_en_val, ans_ar_val)
        sig = _sha(norm_key) if norm_key else None
        if sig and db.query(Question.id).filter(Question.signature == sig).first():
            dup_skips += 1
            return True, sh, sig
        return False, sh, sig

    def _persist_text_item(item: dict, diff_cur: str) -> bool:
        nonlocal created
        try:
            stem_en_val = (item.get("stem_en") or "").strip()
            stem_ar_val = (item.get("stem_ar") or "").strip()
            ans_en_val = (item.get("answer_en") or "").strip()
            ans_ar_val = (item.get("answer_ar") or "").strip()
            hint_val = (item.get("hint") or hint or "").strip()

            if not stem_ar_val:
                return False

            exists, sh, sig = _already_exists(stem_en_val, stem_ar_val, "text", ans_en_val, ans_ar_val)
            if exists:
                return False

            q = Question(
                category_id=cat.id,
                stem_en=stem_en_val or None,
                stem_ar=stem_ar_val,
                answer_en=ans_en_val or None,
                answer_ar=ans_ar_val or None,
                hint=hint_val or None,
                subtopic=(subtopic.strip() or None),
                difficulty=diff_cur,
                region=(region or "saudi"),
                saudi_ratio=float(saudi_ratio) if saudi_ratio is not None else 1.0,
                current_affairs=bool(current_affairs),
                question_mode="TEXT",
                question_type="text",
                media_type=None,
                game_type=(game_type or ("mcq_selection" if is_mcq else "text_input")),
                answer_type=("text_input" if (game_type == "be_loud") else (game_type or ("mcq_selection" if is_mcq else "text_input"))),
                status=("needs_factcheck" if current_affairs else "draft"),
                signature=sig if sig else None,
                stem_hash=sh,
            )

            if is_mcq:
                opts_en = [(item.get(f"option{i}_en") or "").strip() for i in range(1, 5)]
                opts_ar = [(item.get(f"option{i}_ar") or "").strip() for i in range(1, 5)]
                if not all(opts_ar) or not all(opts_en):
                    return False
                try:
                    ci = int(item.get("correct_option_index") or 0)
                except Exception:
                    return False
                if ci not in (0, 1, 2, 3):
                    return False
                q.option1_en, q.option2_en, q.option3_en, q.option4_en = opts_en
                q.option1_ar, q.option2_ar, q.option3_ar, q.option4_ar = opts_ar
                q.correct_option_index = ci
                # keep answer aligned
                q.answer_en = opts_en[ci]
                q.answer_ar = opts_ar[ci]

            db.add(q)
            _db_retry(lambda: db.flush(), label="ai_generate.flush")
            created_ids.append(q.id)
            created += 1
            return (True, "inserted")
        except Exception as e:
            errors.append(str(e))
            return False

    def _generate_text_batch(diff_cur: str, n_needed: int) -> list[dict]:
        nonlocal parse_failures
        variation_key = str(int.from_bytes(os.urandom(4), "big"))
        n_req = max(1, min(int(n_needed), 12))
        avoid = ("AVOID REPEATS (hard rule):\n" + recent_block + "\n") if recent_block else ""
        learn = ("MODERATOR STYLE GUIDANCE (do not copy; use as guidance):\n" + learning_block + "\n") if learning_block else ""

        if is_mcq:
            prompt = (
                f"{common_rules}{difficulty_rules.get(diff_cur, '')}"
                + topic_line + topic_ar_line
                + region_line + saudi_ratio_line + current_affairs_line
                + hint_line + subtopic_line
                + extra_rules
                + avoid + learn
                + f"VariationKey: {variation_key}\n"
                + f"Generate a STRICT JSON array of length {n_req}. Each item keys:\n"
                + "{stem_en, stem_ar, option1_en, option2_en, option3_en, option4_en, option1_ar, option2_ar, option3_ar, option4_ar, correct_option_index, answer_en, answer_ar, hint}\n"
                + "Constraints:\n"
                + "- Exactly 4 options.\n"
                + "- Exactly 1 correct.\n"
                + "- Distractors must be close and same type.\n"
                + "- Do NOT put the correct answer inside the stem.\n"
                + "- answer_en/answer_ar must equal the correct option.\n"
            )
        else:
            prompt = (
                f"{common_rules}{difficulty_rules.get(diff_cur, '')}"
                + topic_line + topic_ar_line
                + region_line + saudi_ratio_line + current_affairs_line
                + hint_line + subtopic_line
                + extra_rules
                + avoid + learn
                + f"VariationKey: {variation_key}\n"
                + f"Generate a STRICT JSON array of length {n_req}. Each item keys:\n"
                + "{stem_en, stem_ar, answer_en, answer_ar, hint}\n"
                + "Constraints:\n"
                + "- Answer must be a single concrete phrase.\n"
                + "- Do NOT include the answer inside the stem.\n"
            )

        try:
            raw = _llm(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                temperature=0.85 if diff_cur in ("hard", "expert") else 0.75,
            )
            out = _parse_json(raw)
            # Normalize common wrapper shapes: {"items":[...]}, {"intents":[...]}, {"data":[...]}
            if isinstance(out, list):
                return out
            if isinstance(out, dict):
                for k in ("items", "intents", "data", "results"):
                    v = out.get(k)
                    if isinstance(v, list):
                        return v
            # If model returns a single object intent, wrap it
            if isinstance(out, dict) and "media_search_query" in out:
                return [out]
            # Unexpected shape -> treat as no intents
            errors.append(f"Intent parse not list ({diff_cur}): {type(out).__name__}")
            return []
        except Exception as e:
            parse_failures += 1
            errors.append(f"LLM/JSON error ({diff_cur}): {e}")
            return []

    # -----------------------------
    # TEXT mode generation
    # -----------------------------
    def _run_text_generation(diffs: list[str], per_diff: int) -> int:
        inserted = 0
        MAX_ATTEMPTS = 8
        for diff_cur in diffs:
            need = per_diff
            attempts = 0
            while need > 0 and attempts < MAX_ATTEMPTS:
                attempts += 1
                batch = _generate_text_batch(diff_cur, n_needed=min(need * 2, 12))
                for item in batch:
                    if need <= 0:
                        break
                    if _persist_text_item(item, diff_cur):
                        inserted += 1
                        need -= 1
                # keep looping until need satisfied or attempts exhausted
            if need > 0:
                errors.append(f"insufficient_unique_text_{diff_cur}:{per_diff-need}/{per_diff}")
        _db_retry(lambda: db.commit(), label="ai_generate.commit.text")
        return inserted

    # -----------------------------
    # MEDIA mode generation (with downgrade-to-text)
    # -----------------------------
    def _run_media_generation(diffs: list[str], per_diff: int) -> int:
        nonlocal media_failures, parse_failures
        inserted = 0
        from agents.media_agent import collect_media_candidates

        MAX_ATTEMPTS = 8

        def _generate_intents(diff_cur: str, n_needed: int) -> list[dict]:
            nonlocal parse_failures
            variation_key = str(int.from_bytes(os.urandom(4), 'big'))
            n_req = max(1, min(int(n_needed), 12))
            avoid = ('AVOID REPEATS (hard rule):\n' + recent_block + '\n') if recent_block else ''
            learn = ('MODERATOR STYLE GUIDANCE (do not copy; use as guidance):\n' + learning_block + '\n') if learning_block else ''

            intent_prompt = (
                f"{common_rules}{difficulty_rules.get(diff_cur, '')}"
                + topic_line + topic_ar_line
                + region_line + saudi_ratio_line + current_affairs_line
                + hint_line + subtopic_line
                + extra_rules + media_intent_extra
                + avoid + learn
                + f"VariationKey: {variation_key}\n"
                + f"MEDIA MODE. media_type={media_type}. difficulty={diff_cur}.\n"
                + f"Return STRICT JSON array of length {n_req} where each item has:\n"
                + "{media_search_query, expected_answer_en, expected_answer_ar, question_style, hint}\n"
                + "question_style must be one of: logo_identification, picture_identification, audio_identification, video_identification, youtube_identification.\n"
                + "expected_answer must be a concrete name.\n"
                + "media_search_query MUST be specific, not generic.\n"
            )

            def _normalize_intent_output(out_obj):
                # returns list[dict] or []
                if isinstance(out_obj, list):
                    return out_obj
                if isinstance(out_obj, dict):
                    for k in ('items', 'intents', 'data', 'results'):
                        v = out_obj.get(k)
                        if isinstance(v, list):
                            return v
                    if 'media_search_query' in out_obj:
                        return [out_obj]
                return []

            def _fallback_intents() -> list[dict]:
                """Offline-safe intent fallback when LLM is unavailable (quota/rate-limit) or returns unusable JSON.

                Must always return a non-empty list with non-empty expected_answer_en so MEDIA generation can proceed
                (even if media is not found, we still insert Q/A with url=None).
                """
                style_map = {
                    'picture': 'picture_identification',
                    'logo': 'logo_identification',
                    'audio': 'audio_identification',
                    'video': 'video_identification',
                    'youtube': 'youtube_identification',
                }
                qstyle = style_map.get(media_type, 'picture_identification')

                base_topic = (getattr(cat, 'name_en', None) or getattr(cat, 'name_ar', None) or 'topic').strip()
                # Heuristic: treat Arabic recipe title as recipe
                topic_l = (base_topic or '').lower()
                is_recipe = is_recipe_category or ('recipe' in topic_l) or ('cook' in topic_l) or ('وصفة' in (getattr(cat, 'name_ar', '') or ''))

                # Simple curated pool for recipe/cooking
                recipe_pool = [
                    ('Kabsa', 'كبسة'),
                    ('Mandi', 'مندي'),
                    ('Jareesh', 'جريش'),
                    ('Harees', 'هريس'),
                    ('Mutabbaq', 'مطبق'),
                    ('Saleeg', 'سليق'),
                    ('Shawarma', 'شاورما'),
                    ('Falafel', 'فلافل'),
                    ('Hummus', 'حمص'),
                    ('Fattoush', 'فتوش'),
                    ('Tabbouleh', 'تبولة'),
                    ('Kunafa', 'كنافة'),
                ]

                intents: list[dict] = []
                for i in range(max(1, n_req)):
                    if is_recipe:
                        en, ar = recipe_pool[(i + int(variation_key) % len(recipe_pool)) % len(recipe_pool)]
                        # Prefer ingredient-style searches for better "raw materials" images
                        q = f"ingredients for {en} flat lay"
                    else:
                        # Generic fallback: use the topic name as a concrete answer to satisfy validation.
                        en = (getattr(cat, 'name_en', None) or 'Topic').strip()
                        ar = (getattr(cat, 'name_ar', None) or en).strip()
                        q = f"{en} reference image"

                    intents.append({
                        'media_search_query': q,
                        'expected_answer_en': en,
                        'expected_answer_ar': ar or en,
                        'question_style': qstyle,
                        'hint': '',
                    })
                return intents
            try:
                raw = _llm(
                    [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': intent_prompt}],
                    temperature=0.7,
                )
                out = _parse_json(raw)
                intents = _normalize_intent_output(out)
                if intents:
                    return intents

                # Second chance: ask for ONE object only (more robust)
                single_prompt = intent_prompt.replace(f"STRICT JSON array of length {n_req}", 'ONE STRICT JSON object')
                raw2 = _llm(
                    [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': single_prompt}],
                    temperature=0.6,
                )
                out2 = _parse_json(raw2)
                intents2 = _normalize_intent_output(out2)
                if intents2:
                    return intents2

                parse_failures += 1
                errors.append(f"Intent parse empty ({diff_cur})")
                logger.warning(f"ai_generate MEDIA: no intents from LLM diff={diff_cur} raw_len={len(str(raw))}")
                return _fallback_intents()
            except Exception as e:
                parse_failures += 1
                errors.append(f"Media intent error ({diff_cur}): {e}")
                logger.warning(f"ai_generate MEDIA: intent exception diff={diff_cur} err={e!r}")
                return _fallback_intents()

        def _persist_media_question(intent: dict, diff_cur: str) -> tuple[bool, str]:
            nonlocal created, media_failures, dup_skips, parse_failures
            q_hint = (intent.get("hint") or hint or "").strip()
            query = (intent.get("media_search_query") or "").strip()
            exp_en = (intent.get("expected_answer_en") or "").strip()
            exp_ar = (intent.get("expected_answer_ar") or "").strip() or exp_en

            if not query:
                query = (getattr(cat, "name_en", None) or getattr(cat, "name_ar", None) or "topic").strip()
            if not exp_en:
                exp_en = (getattr(cat, "name_en", None) or getattr(cat, "name_ar", None) or "Topic").strip() or "Topic"
            if not exp_ar:
                exp_ar = exp_en

            if is_recipe_category and "ingredient" not in query.lower():
                query = (query + " ingredients flat lay").strip()

            # Collect candidates (per-source caps handled in media_agent)
            try:
                cands = collect_media_candidates(query, limit=12, media_type=media_type)
            except Exception as e:
                # keep going; we will still insert question with url=None
                media_failures += 1
                cands = []
                errors.append(f"Media candidate fetch error ({diff_cur}): {type(e).__name__}")

            candidate_dicts = [c for c in (cands or []) if isinstance(c, dict)]
            # Phase-2 (generic) mismatch reduction:
            # - Prefer highest-score candidate, not the first
            # - For logo media, avoid stock-photo sources by default (common mismatch source)
            allow_stock_for_logo = (os.getenv("ALLOW_STOCK_FOR_LOGO", "0").strip() == "1")
            filtered = []
            for c in candidate_dicts:
                u = (c.get("url") or "").strip()
                if not u:
                    continue
                src = (c.get("source") or "unknown").strip().lower()
                if (media_type or "").lower() == "logo" and (not allow_stock_for_logo):
                    if src in ("pexels", "unsplash"):
                        continue
                filtered.append(c)

            # Sort by best score first; fallback to original order if scores missing.
            def _score(c):
                try:
                    return float(c.get("score") or 0.0)
                except Exception:
                    return 0.0
            filtered = sorted(filtered, key=_score, reverse=True)
            urls = [c.get("url") for c in filtered if c.get("url")]
            if not urls:
                # No media found; still create the question (url=None) so it can be enriched later
                media_failures += 1
                selected_url = None
            else:
                # Do not auto-select final media by default (prevents mismatches).
                # Set AUTO_SELECT_MEDIA=1 to keep old behavior.
                auto_select = (os.getenv('AUTO_SELECT_MEDIA', '0').strip() == '1')
                selected_url = urls[0] if auto_select else None

# Generate question based on selected_url + expected answer
            variation_key = str(int.from_bytes(os.urandom(4), "big"))
            avoid = ("AVOID REPEATS (hard rule):\n" + recent_block + "\n") if recent_block else ""
            learn = ("MODERATOR STYLE GUIDANCE (do not copy; use as guidance):\n" + learning_block + "\n") if learning_block else ""

            if is_mcq:
                q_prompt = (
                    f"{common_rules}{difficulty_rules.get(diff_cur, '')}"
                    + f"Create ONE media-based identification question. media_type={media_type}.\n"
                    + "Do NOT mention any URL or link in the stem. Refer only to 'the image/audio/video'.\n"
                    + f"Correct answer (English): {exp_en}\n"
                    + f"Correct answer (Arabic): {exp_ar}\n"
                    + f"Hint: {q_hint or '(none)'}\n"
                    + avoid + learn
                    + f"VariationKey: {variation_key}\n"
                    + "Return STRICT JSON object keys:\n"
                    + "{stem_en, stem_ar, option1_en, option2_en, option3_en, option4_en, option1_ar, option2_ar, option3_ar, option4_ar, correct_option_index, answer_en, answer_ar, hint}\n"
                    + "Constraints:\n"
                    + "- Stem must explicitly reference the media (logo/image/audio/video).\n"
                    + "- Options must be close distractors of the same type.\n"
                    + "- Exactly 1 correct.\n"
                )
            else:
                q_prompt = (
                    f"{common_rules}{difficulty_rules.get(diff_cur, '')}"
                    + f"Create ONE media-based identification question. media_type={media_type}.\n"
                    + "Do NOT mention any URL or link in the stem. Refer only to 'the image/audio/video'.\n"
                    + f"Correct answer (English): {exp_en}\n"
                    + f"Correct answer (Arabic): {exp_ar}\n"
                    + f"Hint: {q_hint or '(none)'}\n"
                    + avoid + learn
                    + f"VariationKey: {variation_key}\n"
                    + "Return STRICT JSON object keys:\n"
                    + "{stem_en, stem_ar, answer_en, answer_ar, hint}\n"
                    + "Constraints:\n"
                    + "- Stem must explicitly reference the media.\n"
                    + "- Answer must be concrete identification.\n"
                )

            def _fallback_media_question_item():
                # Offline-safe question generator when LLM is unavailable (quota/rate-limit).
                # Must keep question_type as MEDIA type; url may be None.
                stem_en = "Identify the item shown in the media."
                stem_ar = "حدد العنصر الظاهر في الوسائط."
                # Slightly better for recipes
                if is_recipe_category:
                    stem_en = "Identify the dish shown in the image."
                    stem_ar = "حدد الطبق الظاهر في الصورة."
                item = {
                    "stem_en": stem_en,
                    "stem_ar": stem_ar,
                    "answer_en": exp_en,
                    "answer_ar": exp_ar or exp_en,
                    "hint": q_hint or "",
                }
                if is_mcq:
                    # Create MCQ options with one correct answer + distractors
                    pool_en = [exp_en, "Kabsa", "Mandi", "Jareesh", "Harees", "Shawarma", "Falafel", "Hummus"]
                    pool_en = [p for p in pool_en if p and p.strip()]
                    # unique preserving order
                    seen=set(); uniq=[]
                    for p in pool_en:
                        if p.lower() not in seen:
                            seen.add(p.lower()); uniq.append(p)
                    # ensure at least 4
                    while len(uniq) < 4:
                        uniq.append(f"Option {len(uniq)+1}")
                    opts = uniq[:4]
                    # shuffle but keep correct index
                    correct = opts[0]
                    random.shuffle(opts)
                    item["options_en"] = opts
                    item["options_ar"] = [exp_ar if o==correct else (exp_ar if (o==exp_en) else o) for o in opts]
                    item["correct_option_index"] = opts.index(correct)
                return item


            def _fallback_mcq_options(answer_en: str, answer_ar: str, category_id: int):
                # Build 4 MCQ options from existing answers in the same category to avoid blanks.
                # This works offline (no LLM) and is generic across all categories.
                ans_en = (answer_en or '').strip() or 'Answer'
                ans_ar = (answer_ar or '').strip() or 'الإجابة'

                # Collect candidate distractors from DB (same category, other answers)
                pool_en = []
                pool_ar = []
                try:
                    rows = (
                        db.query(Question.answer_en, Question.answer_ar)
                        .filter(Question.category_id == category_id)
                        .filter(Question.answer_en.isnot(None))
                        .filter(Question.answer_en != ans_en)
                        .order_by(Question.id.desc())
                        .limit(50)
                        .all()
                    )
                    for ae, aa in rows:
                        if ae and ae.strip() and ae.strip() not in pool_en:
                            pool_en.append(ae.strip())
                        if aa and aa.strip() and aa.strip() not in pool_ar:
                            pool_ar.append(aa.strip())
                except Exception:
                    pass

                # Ensure we have enough options
                def _fill_pool(pool, base):
                    i = 1
                    while len(pool) < 3:
                        cand = f"{base} {i}".strip()
                        if cand != base and cand not in pool:
                            pool.append(cand)
                        i += 1
                    return pool

                pool_en = _fill_pool(pool_en, ans_en)
                pool_ar = _fill_pool(pool_ar, ans_ar)

                # Compose options with the correct answer included
                opts_en = [ans_en] + pool_en[:3]
                opts_ar = [ans_ar] + pool_ar[:3]

                # Shuffle while keeping track of correct index
                import random
                idxs = list(range(4))
                random.shuffle(idxs)
                opts_en_shuf = [opts_en[i] for i in idxs]
                opts_ar_shuf = [opts_ar[i] for i in idxs]
                correct_index = idxs.index(0)
                return opts_en_shuf, opts_ar_shuf, correct_index

            try:
                q_raw = _llm(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": q_prompt}],
                    temperature=0.7,
                )
                q_item = _parse_json(q_raw)
            except Exception as e:
                # If OpenAI quota is exceeded, fall back to deterministic question generation
                if openai_quota_exceeded or (isinstance(e, RuntimeError) and "OPENAI_QUOTA_EXCEEDED" in str(e)):
                    logger.warning(f"ai_generate MEDIA: LLM unavailable, using fallback question. diff={diff_cur}")
                    q_item = _fallback_media_question_item()
                else:
                    parse_failures += 1
                    errors.append(f"Media question LLM/JSON error ({diff_cur}): {e}")
                    return (False, "persist_failed")

            stem_en_val = (q_item.get("stem_en") or "").strip()
            stem_ar_val = (q_item.get("stem_ar") or "").strip()
            if not _ensure_media_reference(stem_en_val, stem_ar_val, media_type):
                if stem_en_val:
                    stem_en_val = f"In this {media_type}, {stem_en_val}".strip()
                if stem_ar_val:
                    stem_ar_val = f"في هذه {'الصورة' if media_type in ('picture','logo') else 'الوسائط'}, {stem_ar_val}".strip()

            ans_en_val = (q_item.get("answer_en") or exp_en).strip()
            ans_ar_val = (q_item.get("answer_ar") or exp_ar).strip()
            hint_val = (q_item.get("hint") or q_hint or "").strip()

            if not stem_ar_val:
                errors.append(f'Persist fail ({diff_cur}): missing stem_ar')
                return (False, 'persist_failed')

            exists, sh, sig = _already_exists(stem_en_val, stem_ar_val, media_type, ans_en_val, ans_ar_val)
            if exists:
                # Deduplicate by stem hash / signature
                return (False, 'duplicate')

            q = Question(
                category_id=cat.id,
                stem_en=stem_en_val or None,
                stem_ar=stem_ar_val,
                answer_en=ans_en_val or None,
                answer_ar=ans_ar_val or None,
                hint=hint_val or None,
                subtopic=(subtopic.strip() or None),
                difficulty=diff_cur,
                question_mode="MEDIA",
                media_type=media_type,
                question_type=media_type,
                game_type=(game_type or ("mcq_selection" if is_mcq else "text_input")),
                answer_type=("text_input" if (game_type == "be_loud") else (game_type or ("mcq_selection" if is_mcq else "text_input"))),
                region=(region or "saudi"),
                status=("needs_factcheck" if current_affairs else "draft"),
                signature=sig if sig else None,
                stem_hash=sh,
                media_status="REVIEW_REQUIRED",
                media_confidence=0.0,
                media_source=None,
                media_intent_json=json.dumps(intent, ensure_ascii=False),
                url=selected_url,
            )

            if is_mcq:
                opts_en = [(q_item.get(f"option{i}_en") or "").strip() for i in range(1, 5)]
                opts_ar = [(q_item.get(f"option{i}_ar") or "").strip() for i in range(1, 5)]
                ci = None
                try:
                    ci = int(q_item.get("correct_option_index") or 0)
                except Exception:
                    ci = None

                # If LLM didn't provide options (or we are in fallback mode), generate reasonable options from DB.
                if (not all(opts_en)) or (not all(opts_ar)) or (ci not in (0,1,2,3)):
                    opts_en, opts_ar, ci = _fallback_mcq_options(ans_en_val, ans_ar_val, cat.id)

                q.option1_en, q.option2_en, q.option3_en, q.option4_en = opts_en
                q.option1_ar, q.option2_ar, q.option3_ar, q.option4_ar = opts_ar
                q.correct_option_index = int(ci or 0)
                q.answer_en = opts_en[q.correct_option_index]
                q.answer_ar = opts_ar[q.correct_option_index]

            db.add(q)
            _db_retry(lambda: db.flush(), label="ai_generate.flush_media")
            created_ids.append(q.id)
            created += 1

            # Store candidates for review
            _db_retry(lambda: db.query(MediaCandidate).filter(MediaCandidate.question_id == q.id).delete(), label="ai_generate.delete_candidates")
            for k, c in enumerate(candidate_dicts[:12]):
                u = c.get("url")
                if not u:
                    continue
                db.add(
                    MediaCandidate(
                        question_id=q.id,
                        source=(c.get("source") or "search"),
                        url=u,
                        title=c.get("title"),
                        score=float(c.get("score") or max(0.0, 1.0 - (k * 0.05))),
                        meta_json=json.dumps(c, ensure_ascii=False),
                        selected=(u == selected_url),
                    )
                )
            return (True, "inserted")
        for diff_cur in diffs:
            need = per_diff
            attempts = 0
            while need > 0 and attempts < MAX_ATTEMPTS:
                attempts += 1
                intents = _generate_intents(diff_cur, n_needed=min(need * 2, 12))
                if not intents:
                    # LLM may be unavailable (quota/rate-limit). Use deterministic fallback intents
                    # so generation still works offline.
                    errors.append(f"No media intents returned ({diff_cur}); using fallback")
                    logger.warning(f"ai_generate MEDIA: no intents diff={diff_cur}; using fallback")

                    def _fallback_media_intents(n_req: int) -> list[dict]:
                        cat_en = (cat.name_en or cat.name_ar or "").strip()
                        cat_low = cat_en.lower()
                        mt = (media_type or "picture").lower()
                        # Topic pools (generic architecture: keyword-based, not hard-coded IDs)
                        if any(k in cat_low for k in ("recipe", "food", "cook", "dish", "cuisine")):
                            pool = ["Kabsa", "Mandi", "Jareesh", "Harees", "Shawarma", "Falafel", "Hummus", "Fattoush", "Mutabbaq", "Kunafa"]
                            style = "picture_identification"
                            return [
                                {
                                    "media_search_query": f"ingredients for {p} flat lay",
                                    "expected_answer_en": p,
                                    "expected_answer_ar": "",
                                    "question_style": style,
                                    "hint": "",
                                }
                                for p in pool[:n_req]
                            ]
                        if any(k in cat_low for k in ("football", "soccer")):
                            pool = ["Al Ahli Saudi FC logo", "Al Hilal logo", "Al Nassr logo", "Al Ittihad logo", "Saudi Pro League logo", "FIFA logo", "AFC Champions League logo"]
                            style = "logo_identification" if mt == "logo" else "picture_identification"
                            intents = []
                            for p in pool[:n_req]:
                                expected = p.replace(" logo", "").strip()
                                intents.append({
                                    "media_search_query": p,
                                    "expected_answer_en": expected,
                                    "expected_answer_ar": "",
                                    "question_style": style,
                                    "hint": "",
                                })
                            return intents

                        # Generic (works for any category) pools by media_type
                        if mt == "logo":
                            pool = ["Nike logo", "Apple logo", "McDonald's logo", "Starbucks logo", "Adidas logo", "Google logo"]
                            style = "logo_identification"
                        elif mt in ("video", "youtube"):
                            pool = ["volcano eruption", "space launch", "football goal celebration", "ocean waves", "camel race"]
                            style = "video_identification"
                        elif mt in ("audio", "sound", "voice"):
                            pool = ["lion roar sound", "rain sound", "crowd cheering sound", "call to prayer audio", "engine rev sound"]
                            style = "audio_identification"
                        else:
                            pool = ["Eiffel Tower", "Taj Mahal", "Great Wall of China", "Statue of Liberty", "Pyramids of Giza", "Burj Khalifa"]
                            style = "picture_identification"

                        out = []
                        for p in pool[:n_req]:
                            out.append({
                                "media_search_query": p,
                                "expected_answer_en": p,
                                "expected_answer_ar": "",
                                "question_style": style,
                                "hint": "",
                            })
                        return out

                    intents = _fallback_media_intents(min(need * 2, 12))
                    if not intents:
                        continue
                for intent in intents:
                    if need <= 0:
                        break
                    ok, reason = _persist_media_question(intent, diff_cur)
                    if ok:
                        inserted += 1
                        need -= 1
                    else:
                        errors.append(f"Media persist fail ({diff_cur}): {reason} | {intent.get('media_search_query','')}")
                        logger.warning(f"ai_generate MEDIA fail: diff={diff_cur} reason={reason} query={intent.get('media_search_query','')}")
            if need > 0:
                inserted += (per_diff - need) - (per_diff - need)  # no-op; created counter updated globally
        _db_retry(lambda: db.commit(), label="ai_generate.commit.media")
        return inserted

    requested_total = target_per_diff * len(diffs_to_generate)

    # Run according to mode
    if question_mode == "TEXT":
        _run_text_generation(diffs_to_generate, target_per_diff)
    else:
        _run_media_generation(diffs_to_generate, target_per_diff)

    # Audit + summary log
    try:
        logger.info(
            f"AI Generate summary: requested={requested_total}, created={created}, "
            f"dup_skips={dup_skips}, media_failures={media_failures}, parse_failures={parse_failures}, "
            f"mode={question_mode}, media_type={media_type}, difficulty={difficulty}, category_id={cat.id}"
        )
    except Exception:
        pass

    try:
        _log_audit(
            db,
            _current_user(request),
            "ai_generate",
            "Category",
            cat.id,
            after_details={
                "mode": question_mode,
                "media_type": media_type,
                "game_type": game_type,
                "difficulty": difficulty,
                "region": region,
                "saudi_ratio": saudi_ratio,
                "current_affairs": bool(current_affairs),
                "subtopic": subtopic,
                "hint": hint,
                "count": target_per_diff,
                "requested_total": requested_total,
                "created": created,
                "dup_skips": dup_skips,
                "media_failures": media_failures,
                "parse_failures": parse_failures,
                "question_ids": created_ids,
            },
            before_details=None,
        )
    except Exception:
        pass

    # errors count includes duplicates + media failures + parse failures + explicit error items
    errors_total = len(errors) + int(dup_skips) + int(media_failures) + int(parse_failures)
    return RedirectResponse(
        f"/questions?category_id={category_id}&generated={created}&errors={errors_total}",
        status_code=303,
    )


@app.post("/agents/media")
def agents_media(
    request: Request,
    question_id: Optional[int] = Form(None),
    category_id: Optional[int] = Form(None),
    only_missing: Optional[str] = Form("0"),
    limit: Optional[int] = Form(25),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Run the media agent for a single question or a batch.

    Batch selection:
    - Only media-based questions (question_mode == 'MEDIA')
    - If only_missing truthy: only questions with no candidates AND no final media_url.
    """
    ids: List[int] = []
    selection_reason = ""

    # --- selection ---
    try:
        if question_id:
            ids = [int(question_id)]
            selection_reason = f"single:{question_id}"
        else:
            q = db.query(Question).filter(Question.category_id == int(category_id or 0))
            # Media agent can run for any media-capable question type (not limited by status/mode)
            q = q.filter(func.lower(func.coalesce(Question.question_type, "")).in_(["picture","logo","audio","video","youtube"]))
            
            if only_missing in ("1", "true", "yes", "on"):
                mc_any2 = db.query(MediaCandidate.question_id).filter(MediaCandidate.question_id == Question.id).exists()
                q = q.filter(~mc_any2).filter(
                    (Question.url == None) | (func.length(func.coalesce(Question.url, "")) == 0)
                )
                selection_reason = "batch:missing_only"

            else:
                selection_reason = "batch:all_media"

            q = q.order_by(Question.id.desc()).limit(int(limit or 25))
            ids = [row.id for row in q.all()]
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print("[agents_media] selection error:\n", tb)
        try:
            _log_audit(db, user, "run_media_agent_select_error", "system", 0, tb[-2000:])
        except Exception:
            pass
        ids = []

    # --- audit start ---
    try:
        _log_audit(
            db,
            user.username,
            "run_media_agent_start",
            "system",
            0,
            json.dumps(
                {
                    "question_id": question_id,
                    "category_id": category_id,
                    "only_missing": only_missing,
                    "limit": limit,
                    "selected": len(ids),
                    "reason": selection_reason,
                }
            ),
        )
    except Exception:
        pass

    # --- execute ---
    processed = 0
    stats = None
    try:
        if ids:
            stats = run_media_agent_for_question_ids(db, ids)
            processed = int(stats.get("processed", len(ids))) if isinstance(stats, dict) else len(ids)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print("[agents_media] run error:\n", tb)
        try:
            _log_audit(db, user.username, "run_media_agent_error", "system", 0, tb[-2000:])
        except Exception:
            pass

    # --- audit end ---
    try:
        payload = {"selected": len(ids), "processed": processed}
        if isinstance(stats, dict):
            # keep short for UI / DB; errors are truncated.
            payload.update({
                "candidates_created": stats.get("candidates_created"),
                "errors": stats.get("errors", [])[-5:],
            })
        _log_audit(db, user.username, "run_media_agent_done", "system", 0, json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass

    if question_id:
        return RedirectResponse(f"/questions/{int(question_id)}/edit", status_code=303)
    if category_id:
        return RedirectResponse(f"/questions?category_id={int(category_id)}", status_code=303)
    return RedirectResponse("/questions", status_code=303)


@app.post("/agents/factcheck")
def agents_factcheck(
    request: Request,
    db: Session = Depends(get_db),
    question_ids: str = Form(""),
):
    _require_login(request)

    ids: List[int] = []
    if question_ids.strip():
        try:
            ids = [int(x) for x in question_ids.split(",") if x.strip()]
        except Exception:
            ids = []

    if not ids:
        qs = (
            db.query(Question)
            .filter(Question.status == "needs_factcheck")
            .order_by(Question.id.desc())
            .limit(20)
            .all()
        )
        ids = [q.id for q in qs]

    if not ids:
        return RedirectResponse("/agents", status_code=303)

    run_factcheck_agent_for_question_ids(db, ids)

    _log_audit(
        db,
        _current_user(request),
        "run_factcheck_agent",
        "Question",
        None,
        {"question_ids": ids},
    )

    return RedirectResponse("/agents", status_code=303)


@app.post("/agents/validate")
def agents_validate(
    request: Request,
    db: Session = Depends(get_db),
    question_ids: str = Form(""),
):
    _require_login(request)

    ids: List[int] = []
    if question_ids.strip():
        try:
            ids = [int(x) for x in question_ids.split(",") if x.strip()]
        except Exception:
            ids = []

    if not ids:
        qs = (
            db.query(Question)
            .filter(Question.status == "needs_validation")
            .order_by(Question.id.desc())
            .limit(20)
            .all()
        )
        ids = [q.id for q in qs]

    if not ids:
        return RedirectResponse("/agents", status_code=303)

    run_validate_agent_for_question_ids(db, ids)

    _log_audit(
        db,
        _current_user(request),
        "run_validate_agent",
        "Question",
        None,
        {"question_ids": ids},
    )

    return RedirectResponse("/agents", status_code=303)


# -------------------------------------------------------------------
# Agents overview
# -------------------------------------------------------------------

@app.get("/agents", response_class=HTMLResponse)
def agents_overview(request: Request, db: Session = Depends(get_db)):
    _require_login(request)

    pending_media = (
        db.query(Question).filter(Question.media_status == "pending").count()
    )
    pending_factcheck = (
        db.query(Question).filter(Question.status == "needs_factcheck").count()
    )
    pending_validation = (
        db.query(Question).filter(Question.status == "needs_validation").count()
    )

    # Media source health panel
    try:
        from agents.media_agent import check_media_source_health
        media_source_health = check_media_source_health()
    except Exception:
        media_source_health = {}

    return templates.TemplateResponse(
        "agents.html",
        {
            "request": request,
            "pending_media": int(pending_media or 0),
            "pending_factcheck": pending_factcheck,
            "pending_validation": pending_validation,
"user": _current_user(request),
        },
    )


# -------------------------------------------------------------------
# Export
# -------------------------------------------------------------------

@app.get("/export", response_class=HTMLResponse)
def export_view(request: Request, db: Session = Depends(get_db)):
    _require_login(request)
    questions = db.query(Question).order_by(Question.id.asc()).all()
    cats = db.query(Category).all()
    cats_map = {c.id: c for c in cats}
    return templates.TemplateResponse(
        "export.html",
        {
            "request": request,
            "questions": questions,
            "categories": cats,
            "cats_map": cats_map,
            "user": _current_user(request),
        },
    )





@app.get("/export/download")
def export_download(
    request: Request,
    category_id: Optional[int] = None,
    question_type: str = "",
    db: Session = Depends(get_db),
):
    """
    Download approved questions as JSON for the main Midan app.

    Rules:
    - Only questions with media_status == 'APPROVED' are exported
    - Optional filters: category_id, question_type
    - Handles AI-generated images that may be stored as /static/... or data:image/... base64 URIs
    """
    _require_login(request)

    q = db.query(Question).filter(Question.media_status == "APPROVED")

    if category_id:
        q = q.filter(Question.category_id == int(category_id))

    qt = (question_type or "").strip().lower()
    if qt:
        q = q.filter(func.lower(func.coalesce(Question.question_type, "")) == qt)

    questions = q.order_by(Question.id.asc()).all()

    def _q_to_dict(x: Question) -> dict:
        url = (getattr(x, "url", None) or getattr(x, "media_url", None) or "").strip()
        is_base64 = url.startswith("data:image/")
        return {
            "id": x.id,
            "category_id": x.category_id,
            "subtopic": getattr(x, "subtopic", None),
            "difficulty": getattr(x, "difficulty", None),
            "question_type": getattr(x, "question_type", None),
            "game_type": getattr(x, "game_type", None) if hasattr(x, "game_type") else getattr(x, "answer_type", None),
            "answer_type": getattr(x, "answer_type", None),
            "question_mode": getattr(x, "question_mode", None),
            "stem_en": getattr(x, "stem_en", None),
            "stem_ar": getattr(x, "stem_ar", None),
            "answer_en": getattr(x, "answer_en", None),
            "answer_ar": getattr(x, "answer_ar", None),
            "hint": getattr(x, "hint", None),
            "option1_en": getattr(x, "option1_en", None),
            "option2_en": getattr(x, "option2_en", None),
            "option3_en": getattr(x, "option3_en", None),
            "option4_en": getattr(x, "option4_en", None),
            "option1_ar": getattr(x, "option1_ar", None),
            "option2_ar": getattr(x, "option2_ar", None),
            "option3_ar": getattr(x, "option3_ar", None),
            "option4_ar": getattr(x, "option4_ar", None),
            "correct_option_index": getattr(x, "correct_option_index", None),
            # Media fields
            "media_type": getattr(x, "media_type", None) or getattr(x, "question_type", None),
            "media_url": url,
            "media_is_base64": is_base64,
            "media_source": getattr(x, "media_source", None),
            "media_selected_source": getattr(x, "media_selected_source", None),
            "media_selected_score": getattr(x, "media_selected_score", None),
            "media_status": getattr(x, "media_status", None),
            # Misc
            "region": getattr(x, "region", None),
            "saudi_ratio": getattr(x, "saudi_ratio", None),
            "current_affairs": int(getattr(x, "current_affairs", 0) or 0),
            "created_at": str(getattr(x, "created_at", "")) if getattr(x, "created_at", None) else None,
            "updated_at": str(getattr(x, "updated_at", "")) if getattr(x, "updated_at", None) else None,
        }

    payload = {"exported_at": datetime.datetime.utcnow().isoformat() + "Z", "count": len(questions), "items": [_q_to_dict(x) for x in questions]}

    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    fname = "midan_export.json" if not category_id else f"midan_export_category_{int(category_id)}.json"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return Response(content=data, media_type="application/json; charset=utf-8", headers=headers)


# ---------------------------
# Lightweight JSON APIs (UI helpers)
# ---------------------------

def _safe_split_lines(text: str) -> List[str]:
    out = []
    for line in (text or "").splitlines():
        t = line.strip(" -•\t").strip()
        if not t:
            continue
        if len(t) > 80:
            t = t[:80].strip()
        if t not in out:
            out.append(t)
    return out[:20]

def _suggest_subtopics_ai(topic: str) -> List[str]:
    topic = (topic or "").strip()
    if not topic:
        return []
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or ""
    if not api_key:
        # deterministic fallback if no key configured
        base = re.sub(r"\s+", " ", topic)
        return [f"{base} — characters", f"{base} — locations", f"{base} — objects", f"{base} — plot details", f"{base} — quotes"]
    try:
        client = OpenAI(api_key=api_key)
        msg = (
            "Generate 12 concise trivia subtopics (2-4 words each) for the topic below.\n"
            "Return as newline-separated items only. No numbering.\n"
            f"TOPIC: {topic}"
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": msg}],
            temperature=0.4,
        )
        text = resp.choices[0].message.content or ""
        items = _safe_split_lines(text)
        return items[:12]
    except Exception:
        return []

@app.get("/api/subtopics/suggest")
def api_subtopics_suggest(topic: str = ""):
    items = _suggest_subtopics_ai(topic)
    return JSONResponse({"items": items})

@app.get("/api/categories/{category_id}/subtopics")
def api_category_subtopics(category_id: int, db: Session = Depends(get_db)):
    cat = db.query(Category).filter(Category.id == int(category_id)).first()
    if not cat:
        return JSONResponse({"items": []})

    rows = (
        db.query(Question.subtopic)
        .filter(Question.category_id == cat.id)
        .filter(Question.subtopic.isnot(None))
        .all()
    )
    existing: List[str] = []
    for (s,) in rows:
        s = (s or "").strip()
        if s and s not in existing:
            existing.append(s)

    if cat.subtopic and cat.subtopic.strip() and cat.subtopic.strip() not in existing:
        existing.insert(0, cat.subtopic.strip())

    if not existing:
        topic = cat.name_en or cat.name_ar or ""
        existing = _suggest_subtopics_ai(topic)

    return JSONResponse({"items": existing[:20]})

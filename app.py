from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
import os
import json
import hashlib
import re
import datetime
from typing import List, Optional, Dict

from fastapi import FastAPI, Request, Form, Depends, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from openai import OpenAI

from database import (
    SessionLocal,
    init_db,
    Category,
    Question,
    MediaCandidate,
    AuditLog,
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

os.makedirs(ORIGINAL_MEDIA_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET_KEY", "change-me-please"),
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
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
):
    """Write an audit record.

    Backwards-compatible with earlier call-sites:
    - _log_audit(db, user, action, entity, id, after_dict)
    - _log_audit(db, user, action, entity, id, after_dict, before_dict)
    """
    try:
        entry = AuditLog(
            actor=user or "system",
            message=action,
            entity=entity,
            entity_id=entity_id,
            before_json=(json.dumps(before_details, ensure_ascii=False) if before_details is not None else None),
            after_json=(json.dumps(after_details, ensure_ascii=False) if after_details is not None else None),
            created_at=datetime.datetime.utcnow(),
        )
        db.add(entry)
        db.commit()
    except Exception:
        # Audit logging must never break core product flows
        db.rollback()
        return


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

    # Media statuses are standardized as: PENDING / REVIEW_REQUIRED / APPROVED
    pending_media = db.query(Question).filter(Question.media_status == "PENDING").count()
    media_review = db.query(Question).filter(Question.media_status == "REVIEW_REQUIRED").count()
    media_approved = db.query(Question).filter(Question.media_status == "APPROVED").count()

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

        c_media_approved = db.query(Question).filter(Question.category_id == c.id, Question.media_status == "APPROVED").count()
        c_media_pending = db.query(Question).filter(Question.category_id == c.id, Question.media_status == "PENDING").count()
        c_media_review = db.query(Question).filter(Question.category_id == c.id, Question.media_status == "REVIEW_REQUIRED").count()

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
            "pending_media": pending_media,
            "media_review": media_review,
            "media_approved": media_approved,
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
        created_at=datetime.datetime.utcnow(),
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
    q_type: str = Form("text"),
    answer_type: str = Form("text"),
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
    upload_media: UploadFile = File(None),
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
    q.question_type = (q_type or "text").strip()
    q.answer_type = (answer_type or "text").strip()
    q.region = (region or "saudi").strip()
    q.saudi_ratio = float(saudi_ratio) if saudi_ratio is not None else 1.0
    q.current_affairs = bool(current_affairs)
    q.subtopic = (subtopic or "").strip()
    q.hint = (hint or "").strip()
    q.status = (status or "draft").strip()
    q.media_query = (media_query or "").strip()

    # Media URL: treat this as the final/canonical media URL. Do not blank existing URL if empty.
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
    if upload_media is not None and getattr(upload_media, "filename", ""):
        uploads_dir = Path("static") / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", upload_media.filename)
        fname = f"q{q.id}_{safe_name}"
        dest = uploads_dir / fname
        with dest.open("wb") as f:
            f.write(upload_media.file.read())
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
        "update",
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
    answer_type: str = Form("mcq_selection"),  # mcq_selection or text_input
    count: int = Form(5),
    subtopic: str = Form(""),
    hint: str = Form(""),
):
    """Production AI generation.

    Rules enforced:
    - No interpretive/philosophical questions.
    - Two modes: TEXT vs MEDIA.
    - Two answer formats: MCQ (1 correct + 3 close distractors) or TEXT (single exact answer).
    - Difficulty must be adhered to by explicit constraints.
    """
    _require_login(request)

    cat = db.query(Category).get(category_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    question_mode = (question_mode or "TEXT").upper().strip()
    if question_mode not in ("TEXT", "MEDIA"):
        question_mode = "TEXT"

    media_type = (media_type or "picture").lower().strip()
    if media_type not in ("logo", "picture", "audio", "video", "youtube"):
        media_type = "picture"

    # Backward compatibility: if older UI sends question_type
    form = request._form if hasattr(request, "_form") else None

    difficulty = (difficulty or "medium").lower().strip()
    if difficulty not in ("easy", "medium", "hard", "expert"):
        difficulty = "medium"

    answer_type = (answer_type or "mcq_selection").lower().strip()
    is_mcq = answer_type.startswith("mcq")

    # --- prompts ---
    common_rules = (
        "STRICT RULES (must comply):\n"
        "1) Questions must be factual and have a single clear correct answer.\n"
        "2) DO NOT generate symbolic, philosophical, interpretive, or opinion-based questions.\n"
        "3) Avoid 'what does it mean/represent/symbolize' style.\n"
        "4) The hint is a HARD constraint; follow it.\n"
        "5) Difficulty must be strictly adhered to.\n"
    )

    difficulty_rules = {
        "easy": (
            "- Easy: direct identification or simple facts.\n"
            "- Use obvious cues.\n"
        ),
        "medium": (
            "- Medium: still factual, but avoid extremely easy prompts (e.g., main character names).\n"
            "- Prefer specific details and close distractors for MCQ.\n"
        ),
        "hard": (
            "- Hard: factual but less obvious; require specific detail recognition.\n"
            "- Avoid extremely obvious examples; prefer less-common but verifiable entities.\n"
            "- Distractors must be very close and plausible.\n"
        ),
        "expert": """
- Expert: factual, precise, single-answer; avoid ambiguity or interpretive/philosophical questions.
- DO NOT ask direct ID questions for very famous items (e.g., "What brand is this?" for Coca-Cola/Apple/Google; or "Name this Disney movie" for Frozen/Encanto).
- Instead, target deep but verifiable details: plot beats, scene-specific objects, minor/supporting characters, locations, magical items, slogans/taglines, release-year within a tight range, etc.
- The correct answer must be clearly derivable from the prompt/media, but not obvious at a glance.
- For MCQ: use 3 close distractors from the same universe/franchise/industry and same era; avoid random/unrelated options.
- Never include the answer text visibly inside the media (watermarks, text overlays, subtitles, obvious labels).
""",

    }

    topic_line = f"Category (topic): {cat.name_en or cat.name_ar or '(unnamed)'}\n"
    topic_ar_line = f"Category Arabic: {cat.name_ar or cat.name_en or '(unnamed)'}\n"
    region_line = f"Region scope: {region or 'saudi'}\n"
    saudi_ratio_line = f"Saudi ratio (0..1): {saudi_ratio}\n"
    current_affairs_line = f"Current affairs: {'YES' if current_affairs else 'NO'}\n"
    hint_line = f"Hint/Prompt: {hint.strip() if hint else '(none)'}\n"
    subtopic_line = f"Subtopic constraint: {subtopic.strip() if subtopic else '(none)'}\n"

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
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=messages,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def _search_wikimedia(query: str, limit: int = 10):
        try:
            from utils.wikipedia_tools import search_commons_images
            return search_commons_images(query=query, limit=limit) or []
        except Exception:
            return []

    created = 0
    created_ids = []
    errors = []
    items = []

    # ----------------------------
    # TEXT MODE
    # ----------------------------
    if question_mode == "TEXT":
        if is_mcq:
            prompt = (
                f"{common_rules}{difficulty_rules[difficulty]}"
                + topic_line
                + topic_ar_line
                + region_line + saudi_ratio_line + current_affairs_line
                + hint_line
                + subtopic_line
                + "RELEVANCE: All questions MUST be strictly about the Category/Topic. Subtopic + Hint are hard constraints.\n"
                + "Generate a STRICT JSON array of length "
                + str(count)
                + " where each item has:\n"
                + "{stem_en, stem_ar, option1_en, option2_en, option3_en, option4_en, "
                + "option1_ar, option2_ar, option3_ar, option4_ar, correct_option_index, "
                + "answer_en, answer_ar, hint}\n"
                + "Constraints for MCQ:\n"
                + "- Exactly 4 options.\n"
                + "- Exactly 1 correct option.\n"
                + "- Distractors must be close and of the same type/category.\n"
                + "- Do NOT put the correct answer inside the question stem.\n"
                + "- answer_en/answer_ar must equal the correct option text.\n"
            )
        else:
            prompt = (
                f"{common_rules}{difficulty_rules[difficulty]}"
                + topic_line
                + topic_ar_line
                + region_line + saudi_ratio_line + current_affairs_line
                + hint_line
                + subtopic_line
                + "RELEVANCE: All questions MUST be strictly about the Category/Topic. Subtopic + Hint are hard constraints.\n"
                + "Generate a STRICT JSON array of length "
                + str(count)
                + " where each item has:\n"
                + "{stem_en, stem_ar, answer_en, answer_ar, hint}\n"
                + "Constraints:\n"
                + "- The answer must be a single concrete phrase (not interpretive).\n"
                + "- Do NOT include the answer inside the question stem.\n"
            )

        raw = ""
        try:
            raw = _llm(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.35,
            )
            items = _parse_json(raw)
        except Exception as e:
            errors.append(f"LLM/JSON parse error: {e}")
            items = []

        # Save
        for item in items:
            try:
                stem_en_val = (item.get("stem_en") or "").strip()
                stem_ar_val = (item.get("stem_ar") or "").strip()
                ans_en_val = (item.get("answer_en") or "").strip()
                ans_ar_val = (item.get("answer_ar") or "").strip()
                hint_val = (item.get("hint") or hint or "").strip()

                if not stem_ar_val:
                    continue

                norm_key = _normalize_question_text(stem_en_val, stem_ar_val, ans_en_val, ans_ar_val)
                sig = _sha(norm_key) if norm_key else None
                if sig:
                    existing = db.query(Question).filter(Question.signature == sig).first()
                    if existing:
                        continue

                q = Question(
                    category_id=cat.id,
                    stem_en=stem_en_val or None,
                    stem_ar=stem_ar_val,
                    answer_en=ans_en_val or None,
                    answer_ar=ans_ar_val or None,
                    hint=hint_val or None,
                    subtopic=(subtopic.strip() or None),
                    difficulty=difficulty,
                    region=(region or "saudi"),
                    saudi_ratio=float(saudi_ratio) if saudi_ratio is not None else 1.0,
                    current_affairs=bool(current_affairs),
                    question_mode=(question_mode or "TEXT"),
                    question_type=("media" if (question_mode or "TEXT") == "MEDIA" else "text"),
                    media_type=(media_type if (question_mode or "TEXT") == "MEDIA" else None),
                    answer_type=("mcq_selection" if is_mcq else "text_input"),
                    status=("needs_factcheck" if current_affairs else "draft"),
                    signature=sig if sig else None,
                )

                if is_mcq:
                    # validate options
                    opts_en = [(item.get(f"option{i}_en") or "").strip() for i in range(1,5)]
                    opts_ar = [(item.get(f"option{i}_ar") or "").strip() for i in range(1,5)]
                    if not all(opts_ar) or not all(opts_en):
                        continue
                    try:
                        ci = int(item.get("correct_option_index") or 0)
                    except Exception:
                        ci = 0
                    if ci not in (0,1,2,3):
                        continue
                    # assign
                    q.option1_en, q.option2_en, q.option3_en, q.option4_en = opts_en
                    q.option1_ar, q.option2_ar, q.option3_ar, q.option4_ar = opts_ar
                    q.correct_option_index = ci

                db.add(q)
                db.flush()  # ensure q.id available
                created_ids.append(q.id)
                created += 1
            except Exception as e:
                errors.append(str(e))
                continue

        db.commit()

        # Auto-fetch media candidates for MEDIA mode (non-blocking)
        if question_mode == "MEDIA" and created_ids:
            try:
                from agents.media_agent import run_media_agent_for_question_ids
                run_media_agent_for_question_ids(db, created_ids)
            except Exception:
                pass

        return RedirectResponse(f"/questions?category_id={category_id}&generated={created}&errors={len(errors)}", status_code=303)

    # ----------------------------
    # MEDIA MODE (media-first)
    # ----------------------------

    # Media intent prompt: must yield concrete identification targets
    intent_prompt = (
        f"{common_rules}{difficulty_rules[difficulty]}"
        + topic_line
        + topic_ar_line
        + region_line + saudi_ratio_line + current_affairs_line
        + hint_line
        + subtopic_line
        + "RELEVANCE: Intents MUST be strictly about the Category/Topic. Subtopic + Hint are hard constraints.\n"
        + f"MEDIA MODE. media_type={media_type}.\n"
        + "Return STRICT JSON array of length "
        + str(count)
        + " where each item has:\n"
        + "{media_search_query, expected_answer_en, expected_answer_ar, question_style, hint}\n"
        + "question_style must be one of: logo_identification, picture_identification, audio_identification, video_identification, youtube_identification.\n"
        + "expected_answer must be a concrete name (company/brand, movie title, singer, instrument, place, object).\n"
        + "media_search_query MUST be specific (include key nouns), not generic.\n"
    )

    try:
        raw = _llm(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": intent_prompt},
            ],
            temperature=0.25,
        )
        intents = _parse_json(raw)
    except Exception as e:
        errors.append(f"Media intent error: {e}")
        intents = []

    from agents.media_agent import collect_media_candidates

    for intent in intents:
        try:
            q_hint = (intent.get("hint") or hint or "").strip()
            query = (intent.get("media_search_query") or "").strip()
            exp_en = (intent.get("expected_answer_en") or "").strip()
            exp_ar = (intent.get("expected_answer_ar") or "").strip()
            if not exp_ar:
                exp_ar = exp_en  # fallback; better than empty
            if not query or not exp_en:
                continue

            # Get media candidates: prefer Wikimedia for logo/picture
            urls = []
            if media_type in ("logo", "picture"):
                urls = _search_wikimedia(query, limit=12)
            # fallback to generic search
            if not urls and media_type == "picture":
                try:
                    cands = collect_media_candidates(query)
                    urls = []
                    for c in cands:
                        if isinstance(c, dict):
                            u = c.get("url")
                        else:
                            u = None
                        if u:
                            urls.append(u)
                    urls = urls[:12]
                except Exception:
                    urls = []

            if not urls:
                # Create question anyway but mark review_required
                selected_url = None
                media_status = "REVIEW_REQUIRED"
                media_conf = 0.0
                media_src = None
            else:
                candidate_urls = urls
                # Do not auto-select or auto-approve media. User must choose.
                selected_url = None
                media_status = "REVIEW_REQUIRED"
                media_conf = 0.0
                media_src = "wikimedia" if media_type in ("logo","picture") else "search"

            # Question generation prompt (must refer to media)
            if is_mcq:
                q_prompt = (
                    f"{common_rules}{difficulty_rules[difficulty]}"
                    + f"Create ONE media-based identification question. media_type={media_type}.\nFor difficulty=hard/expert: do NOT use extremely obvious household examples; choose less-obvious but verifiable entities.\n"
                    + f"Media URL: {selected_url or '(none)'}\n"
                    + f"Correct answer (English): {exp_en}\n"
                    + f"Correct answer (Arabic): {exp_ar}\n"
                    + f"Hint: {q_hint or '(none)'}\n"
                    + "Return STRICT JSON object with keys:\n"
                    + "{stem_en, stem_ar, option1_en, option2_en, option3_en, option4_en, option1_ar, option2_ar, option3_ar, option4_ar, correct_option_index, answer_en, answer_ar, hint}\n"
                    + "Constraints:\n"
                    + "- Stem must explicitly reference the media (logo/image/audio/video).\n"
                    + "- Options must be close distractors of the same type.\n"
                    + "- Exactly 1 correct.\n"
                    + "- answer_en/answer_ar must equal the correct option.\n"
                )
            else:
                q_prompt = (
                    f"{common_rules}{difficulty_rules[difficulty]}"
                    + f"Create ONE media-based identification question. media_type={media_type}.\nFor difficulty=hard/expert: do NOT use extremely obvious household examples; choose less-obvious but verifiable entities.\n"
                    + f"Media URL: {selected_url or '(none)'}\n"
                    + f"Correct answer (English): {exp_en}\n"
                    + f"Correct answer (Arabic): {exp_ar}\n"
                    + f"Hint: {q_hint or '(none)'}\n"
                    + "Return STRICT JSON object with keys:\n"
                    + "{stem_en, stem_ar, answer_en, answer_ar, hint}\n"
                    + "Constraints:\n"
                    + "- Stem must explicitly reference the media.\n"
                    + "- Answer must be concrete identification.\n"
                )

            q_raw = _llm(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": q_prompt},
                ],
                temperature=0.25,
            )
            q_item = _parse_json(q_raw)

            stem_en_val = (q_item.get("stem_en") or "").strip()
            stem_ar_val = (q_item.get("stem_ar") or "").strip()
            ans_en_val = (q_item.get("answer_en") or exp_en).strip()
            ans_ar_val = (q_item.get("answer_ar") or exp_ar).strip()
            hint_val = (q_item.get("hint") or q_hint or "").strip()

            if not stem_ar_val:
                continue

            norm_key = _normalize_question_text(stem_en_val, stem_ar_val, ans_en_val, ans_ar_val)
            sig = _sha(norm_key) if norm_key else None
            if sig:
                if db.query(Question).filter(Question.signature == sig).first():
                    continue

            q = Question(
                category_id=cat.id,
                stem_en=stem_en_val or None,
                stem_ar=stem_ar_val,
                answer_en=ans_en_val or None,
                answer_ar=ans_ar_val or None,
                hint=hint_val or None,
                subtopic=(subtopic.strip() or None),
                difficulty=difficulty,
                question_mode="MEDIA",
                media_type=media_type,
                question_type=media_type,  # keep legacy field aligned
                answer_type=("mcq_selection" if is_mcq else "text_input"),
                region=(region or "saudi"),
                status=("needs_factcheck" if current_affairs else "draft"),
                signature=sig if sig else None,
                media_status=media_status,
                media_confidence=media_conf,
                media_source=media_src,
                media_intent_json=json.dumps(intent, ensure_ascii=False),
            )

            # set selected media fields
            if selected_url:
                # legacy url field + new fields
                q.url = selected_url
                if hasattr(q, "media_url"):
                    q.media_url = selected_url
                if hasattr(q, "media_query"):
                    q.media_query = query
                if hasattr(q, "media_selected_source"):
                    q.media_selected_source = media_src
                if hasattr(q, "media_selected_score"):
                    q.media_selected_score = str(media_conf)
                if hasattr(q, "media_type"):
                    q.media_type = media_type

            if is_mcq:
                opts_en = [(q_item.get(f"option{i}_en") or "").strip() for i in range(1,5)]
                opts_ar = [(q_item.get(f"option{i}_ar") or "").strip() for i in range(1,5)]
                if not all(opts_ar) or not all(opts_en):
                    continue

                # Determine correct index by matching answer text (more reliable than trusting the model index)
                model_ci = 0
                try:
                    model_ci = int(q_item.get("correct_option_index") or 0)
                except Exception:
                    model_ci = 0

                # Prefer matching by English answer, else expected answer, else model index
                ans_key = (ans_en_val or "").strip().lower()
                exp_key = (exp_en or "").strip().lower()
                correct_idx = -1
                for i_opt, opt in enumerate(opts_en, start=0):
                    if ans_key and opt.lower() == ans_key:
                        correct_idx = i_opt
                        break
                if correct_idx == -1 and exp_key:
                    for i_opt, opt in enumerate(opts_en, start=0):
                        if opt.lower() == exp_key:
                            correct_idx = i_opt
                            break
                if correct_idx == -1 and model_ci in (0,1,2,3):
                    correct_idx = model_ci
                if correct_idx == -1:
                    continue

                q.option1_en, q.option2_en, q.option3_en, q.option4_en = opts_en
                q.option1_ar, q.option2_ar, q.option3_ar, q.option4_ar = opts_ar
                q.correct_option_index = correct_idx

                # Ensure stored answer matches correct option
                q.answer_en = opts_en[correct_idx]
                q.answer_ar = opts_ar[correct_idx]

            db.add(q)
            db.flush()  # allocate q.id for media candidates

            # For MEDIA mode, pre-populate media_candidates so UI is not blank.
            if question_mode == "MEDIA":
                # clear any existing candidates for this question (safety)
                db.query(MediaCandidate).filter(MediaCandidate.question_id == q.id).delete()
                if urls:
                    # score descending by order; mark the first as selected
                    for k, u in enumerate(urls[:12]):
                        db.add(
                            MediaCandidate(
                                question_id=q.id,
                                source=media_src or "search",
                                url=u,
                                title=None,
                                score=float(max(0.0, media_conf - (k * 0.03))),
                                meta_json=None,
                                selected=False,
                            )
                        )

            created += 1

        except Exception as e:
            errors.append(str(e))
            continue

    db.commit()
    return RedirectResponse(f"/questions?category_id={category_id}&generated={created}&errors={len(errors)}", status_code=303)
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
            q = q.filter(Question.question_mode == "MEDIA")

            if only_missing in ("1", "true", "yes", "on"):
                q = q.filter(
                    (Question.media_status == None) | (Question.media_status != "APPROVED")
                ).filter(
                    (Question.url == None) | (func.length(func.coalesce(Question.url, "")) == 0)
                    | (Question.media_candidates_json == None)
                    | (func.length(func.coalesce(Question.media_candidates_json, "")) == 0)
                    | (Question.media_candidates_json == "[]")
                    | (Question.media_status == "REVIEW_REQUIRED")
                    | (Question.media_status == "IN_REVIEW")
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
            _log_audit(db, user.username, "run_media_agent_select_error", "system", 0, tb[-2000:])
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

    return templates.TemplateResponse(
        "agents.html",
        {
            "request": request,
            "pending_media": pending_media,
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
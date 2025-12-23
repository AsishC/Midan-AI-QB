from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
import os
import json
import hashlib
import datetime
from typing import List, Optional, Dict

from fastapi import FastAPI, Request, Form, Depends, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
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


def _require_login(request: Request) -> None:
    if not _current_user(request):
        raise HTTPException(status_code=401, detail="Not authenticated")


def _log_audit(
    db: Session,
    user: Optional[str],
    action: str,
    entity: str,
    entity_id: Optional[int],
    details: Dict,
):
    entry = AuditLog(
        actor=user or "system",
        message=action,
        entity=entity,
        entity_id=entity_id,
        before_json=None,
        after_json=json.dumps(details, ensure_ascii=False),
        created_at=datetime.datetime.utcnow(),
    )
    db.add(entry)
    db.commit()


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
    pending_media = (
        db.query(Question).filter(Question.media_status == "pending").count()
    )
    pending_factcheck = (
        db.query(Question).filter(Question.status == "needs_factcheck").count()
    )
    pending_validation = (
        db.query(Question).filter(Question.status == "needs_validation").count()
    )

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
            "pending_media": pending_media,
            "pending_factcheck": pending_factcheck,
            "pending_validation": pending_validation,
            "recent_audit": recent_audit,
        },
    )


# -------------------------------------------------------------------
# Categories
# -------------------------------------------------------------------

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
    name_en: str = Form(...),
    name_ar: str = Form(""),
    scope: str = Form("global"),
    subtopic: str = Form(""),
    description_en: str = Form(""),
    description_ar: str = Form(""),
    saudi_safe_notes: str = Form(""),
    is_current_affairs: bool = Form(False),
    db: Session = Depends(get_db),
):
    _require_login(request)
    name_en = (name_en or "").strip()
    if not name_en:
        raise HTTPException(status_code=400, detail="Category name (EN) is required")

    existing = (
        db.query(Category)
        .filter(Category.name_en.ilike(name_en))
        .first()
    )
    if existing:
        return RedirectResponse("/categories", status_code=303)

    cat = Category(
        name_en=name_en,
        name_ar=(name_ar or "").strip(),
        scope=(scope or "global").strip(),
        subtopic=(subtopic or "").strip(),
        description_en=(description_en or "").strip(),
        description_ar=(description_ar or "").strip(),
        saudi_safe_notes=(saudi_safe_notes or "").strip(),
        is_current_affairs=bool(is_current_affairs),
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
            "scope": cat.scope,
            "is_current_affairs": cat.is_current_affairs,
        },
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
    name_en: str = Form(...),
    name_ar: str = Form(""),
    scope: str = Form("global"),
    subtopic: str = Form(""),
    description_en: str = Form(""),
    description_ar: str = Form(""),
    saudi_safe_notes: str = Form(""),
    is_current_affairs: bool = Form(False),
    db: Session = Depends(get_db),
):
    _require_login(request)

    cat = db.query(Category).get(category_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    old = {
        "name_en": cat.name_en,
        "scope": cat.scope,
        "subtopic": cat.subtopic,
        "description_en": cat.description_en,
        "is_current_affairs": cat.is_current_affairs,
    }

    cat.name_en = (name_en or "").strip()
    cat.name_ar = (name_ar or "").strip()
    cat.scope = (scope or "global").strip()
    cat.subtopic = (subtopic or "").strip()
    cat.description_en = (description_en or "").strip()
    cat.description_ar = (description_ar or "").strip()
    cat.saudi_safe_notes = (saudi_safe_notes or "").strip()
    cat.is_current_affairs = bool(is_current_affairs)
    db.commit()

    _log_audit(
        db,
        _current_user(request),
        "update",
        "Category",
        cat.id,
        {"old": old, "new": {
            "name_en": cat.name_en,
            "scope": cat.scope,
            "subtopic": cat.subtopic,
            "description_en": cat.description_en,
            "is_current_affairs": cat.is_current_affairs,
        }},
    )

    return RedirectResponse("/categories", status_code=303)


# -------------------------------------------------------------------
# Questions list & detail
# -------------------------------------------------------------------


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
    subtopic: str = Form(""),
    hint: str = Form(""),
    status: str = Form("draft"),
    media_query: str = Form(""),
    media_url: str = Form(""),
    media_type: str = Form(""),
    media_status: str = Form("none"),
    media_selected_source: str = Form(""),
    media_selected_score: str = Form(""),
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
        "media_url": q.media_url,
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
    q.subtopic = (subtopic or "").strip()
    q.hint = (hint or "").strip()
    q.status = (status or "draft").strip()
    q.media_query = (media_query or "").strip()
    q.media_url = (media_url or "").strip()
    q.media_type = (media_type or "").strip()
    q.media_status = (media_status or "").strip()
    q.media_selected_source = (media_selected_source or "").strip()
    q.media_selected_score = (media_selected_score or "").strip()

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
            "media_url": q.media_url,
        }},
    )

    return RedirectResponse(f"/questions/{q.id}/edit", status_code=303)


# -------------------------------------------------------------------
# AI question generation
# -------------------------------------------------------------------

@app.get("/ai/generate", response_class=HTMLResponse)
def ai_generate_form(request: Request, db: Session = Depends(get_db)):
    _require_login(request)
    cats = db.query(Category).order_by(Category.name_en.asc()).all()
    return templates.TemplateResponse(
        "ai_generate.html",
        {"request": request, "categories": cats, "user": _current_user(request)},
    )


@app.post("/ai/generate", response_class=HTMLResponse)
def ai_generate_post(
    request: Request,
    db: Session = Depends(get_db),
    category_id: int = Form(...),
    difficulty: str = Form("medium"),
    question_type: str = Form("text"),
    answer_type: str = Form("mcq_selection"),
    count: int = Form(5),
    subtopic: str = Form(""),
    hint: str = Form(""),
):
    _require_login(request)

    cat = db.query(Category).get(category_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    subtopic_use = (subtopic or "").strip() or (cat.subtopic or "").strip()
    hint_use = (hint or "").strip()
    safety = (cat.saudi_safe_notes or "").strip()

    # Build the system + user prompt for OpenAI
    prompt = (
        f"Generate {count} quiz questions. Category: {cat.name_en}. "
        f"Scope: {cat.scope}. Difficulty: {difficulty}. "
        f"Question type: {question_type}. Answer type: {answer_type}. "
    )
    if subtopic_use:
        prompt += f"Subtopic: {subtopic_use}. "
    if hint_use:
        prompt += f"Hint: {hint_use}. "
    if cat.is_current_affairs:
        prompt += (
            "These are current affairs questions; avoid outdated facts and keep answers verifiable. "
        )
    if safety:
        prompt += safety + " "

    # Required JSON schema for the generator (strict; no markdown, no code fences)
    if answer_type.startswith("mcq"):
        prompt += (
            "Return a STRICT JSON array (no markdown) of objects with keys: "
            "stem_en, stem_ar, answer_en, answer_ar, "
            "option1_en, option2_en, option3_en, option4_en, "
            "option1_ar, option2_ar, option3_ar, option4_ar, "
            "correct_option_index. "
            "For MCQ questions you MUST provide one clearly correct answer and three plausible distractors. "
            "All four options MUST be present in both English and Arabic; never leave Arabic options empty. "
            "Respect the requested difficulty level exactly: for medium and hard questions, avoid extremely easy "
            "facts (like only main character names) and instead ask about deeper plot details, character traits, "
            "supporting characters, locations, magical objects, or notable scenes. "
        )
    else:
        prompt += (
            "Return a STRICT JSON array (no markdown) of objects with keys: "
            "stem_en, stem_ar, answer_en, answer_ar. "
        )

    system_msg = (
        "You are an exam-quality question generator for a Saudi family-safe trivia game. "
        "You MUST follow all instructions exactly. "
        "All Arabic text must be proper Modern Standard Arabic. "
        "Do not include the answer inside the question stem. "
        "The difficulty must strictly match the requested level."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=messages,
            temperature=0.7,
        )
        raw = resp.choices[0].message.content or "[]"
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if "\n" in raw:
                raw = raw.split("\n", 1)[1]
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("AI did not return a list")

    except Exception as e:
        print("[AI ERROR]", e)
        return templates.TemplateResponse(
            "ai_generate.html",
            {
                "request": request,
                "categories": db.query(Category).all(),
                "error": f"AI generation failed: {e}",
                "user": _current_user(request),
            },
            status_code=500,
        )

    created_ids: List[int] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        stem_en_val = (item.get("stem_en") or "").strip()
        stem_ar_val = (item.get("stem_ar") or "").strip()
        ans_en_val = (item.get("answer_en") or "").strip()
        ans_ar_val = (item.get("answer_ar") or "").strip()

        if not stem_en_val and not stem_ar_val:
            continue
        if not ans_en_val and not ans_ar_val:
            continue

        # Basic difficulty sanity-check – reject questions that are too short
        stem_en_len = len(stem_en_val or "")
        if difficulty in ("medium", "hard"):
            min_len = 40 if difficulty == "medium" else 60
            if stem_en_len < min_len:
                # Too simple/short for the requested difficulty – skip
                continue

        # Reject half-filled MCQs that are missing Arabic options
        if answer_type.startswith("mcq"):
            arabic_opts = [
                (item.get("option1_ar") or "").strip(),
                (item.get("option2_ar") or "").strip(),
                (item.get("option3_ar") or "").strip(),
                (item.get("option4_ar") or "").strip(),
            ]
            # If any Arabic option is missing, skip this question
            if not all(arabic_opts):
                continue

        option1_en = (item.get("option1_en") or "").strip()
        option2_en = (item.get("option2_en") or "").strip()
        option3_en = (item.get("option3_en") or "").strip()
        option4_en = (item.get("option4_en") or "").strip()

        option1_ar = (item.get("option1_ar") or "").strip()
        option2_ar = (item.get("option2_ar") or "").strip()
        option3_ar = (item.get("option3_ar") or "").strip()
        option4_ar = (item.get("option4_ar") or "").strip()

        correct_idx = item.get("correct_option_index")
        if answer_type.startswith("mcq"):
            try:
                correct_idx = int(correct_idx)
            except Exception:
                correct_idx = None

        norm_key = _normalize_question_text(stem_en_val, stem_ar_val, ans_en_val, ans_ar_val)
        if norm_key:
            sig = _sha(norm_key)
            existing = (
                db.query(Question)
                .filter(Question.signature == sig)
                .first()
            )
            if existing:
                continue

        q = Question(
            category_id=category_id,
            stem_en=stem_en_val,
            stem_ar=stem_ar_val,
            answer_en=ans_en_val,
            answer_ar=ans_ar_val,
            option1_en=option1_en,
            option2_en=option2_en,
            option3_en=option3_en,
            option4_en=option4_en,
            option1_ar=option1_ar,
            option2_ar=option2_ar,
            option3_ar=option3_ar,
            option4_ar=option4_ar,
            correct_option_index=correct_idx,
            difficulty=difficulty,
            question_type=question_type,
            answer_type=answer_type,
            region=cat.scope or "saudi",
            subtopic=subtopic_use,
            hint=hint_use,
            signature=sig if norm_key else None,
            status="needs_factcheck",
            media_status="pending",
            created_at=datetime.datetime.utcnow(),
        )

        if norm_key:
            q.dedup_key = _sha(norm_key)

        db.add(q)
        db.commit()
        created_ids.append(q.id)

        _log_audit(
            db,
            _current_user(request),
            "create",
            "Question",
            q.id,
            {
                "category_id": category_id,
                "difficulty": difficulty,
                "question_type": question_type,
                "answer_type": answer_type,
            },
        )

    if not created_ids:
        return templates.TemplateResponse(
            "ai_generate.html",
            {
                "request": request,
                "categories": db.query(Category).all(),
                "error": "No valid questions were created – likely all were rejected by validation.",
                "user": _current_user(request),
            },
            status_code=400,
        )

    return RedirectResponse("/questions", status_code=303)


# -------------------------------------------------------------------
# Media agent trigger
# -------------------------------------------------------------------

@app.post("/agents/media")
def agents_media(
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
            .filter(Question.media_status == "pending")
            .order_by(Question.id.desc())
            .limit(20)
            .all()
        )
        ids = [q.id for q in qs]

    if not ids:
        return RedirectResponse("/agents", status_code=303)

    run_media_agent_for_question_ids(db, ids)

    _log_audit(
        db,
        _current_user(request),
        "run_media_agent",
        "Question",
        None,
        {"question_ids": ids},
    )

    return RedirectResponse("/agents", status_code=303)


# -------------------------------------------------------------------
# Fact-check & validate agents
# -------------------------------------------------------------------

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
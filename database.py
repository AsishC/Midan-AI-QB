import os
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    Boolean,
    inspect,
    text,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./midan.db")

# SQLite can hit "database is locked" under concurrent writes (FastAPI threadpool + background agents).
# Mitigations:
# - WAL journal mode (better concurrency)
# - busy_timeout (wait for locks instead of failing immediately)
# - longer driver timeout
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False,
        # seconds: wait on file locks before raising
        "timeout": float(os.getenv("SQLITE_TIMEOUT", "30")),
    }

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    poolclass=StaticPool,
) if DATABASE_URL.startswith('sqlite') else create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """Apply pragmatic defaults for SQLite concurrency & integrity."""
    try:
        cursor = dbapi_connection.cursor()
        # Only applies to sqlite connections.
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        # milliseconds
        cursor.execute(f"PRAGMA busy_timeout={int(float(os.getenv('SQLITE_BUSY_TIMEOUT_MS','5000')))}")
        cursor.close()
    except Exception:
        # If not sqlite or pragma unsupported, ignore.
        pass

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)

    # Arabic-first UX, but allow EN-only categories.
    # Validation in app layer enforces: at least one of name_ar or name_en must be provided.
    name_ar = Column(String(120), nullable=True)
    name_en = Column(String(120), nullable=True)

    description_ar = Column(Text, nullable=True)
    description_en = Column(Text, nullable=True)

    # Optional default subtopic seed (AI can suggest; human can override)
    subtopic = Column(String(120), nullable=True)

    # Used for dedupe / category identity (optional)
    dedup_key = Column(String(80), nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    questions = relationship("Question", back_populates="category", cascade="all, delete-orphan")


class Question(Base):

    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)

    # Arabic is primary in UI; EN is optional
    stem_ar = Column(Text, nullable=False)
    stem_en = Column(Text, nullable=True)

    answer_ar = Column(Text, nullable=True)
    answer_en = Column(Text, nullable=True)

    hint = Column(Text, nullable=True)
    subtopic = Column(String(120), nullable=True)

    difficulty = Column(String(20), default="medium")  # easy/medium/hard/expert
    question_type = Column(String(20), default="text")  # text/picture/audio/video/logo
    answer_type = Column(String(30), default="mcq_selection")  # legacy: mcq_selection/text_input
    # Preferred: game type (MCQ, Text Input, Be Loud, etc.)
    game_type = Column(String(30), default="mcq_selection")
    # Generation mode: TEXT (no media required) vs MEDIA (question depends on media)
    question_mode = Column(String(10), default="TEXT")  # TEXT/MEDIA
    # Media type when question_mode == MEDIA
    media_type = Column(String(20), nullable=True)  # logo/picture/audio/video/youtube
    # Internal control JSON for media-first generation
    media_intent_json = Column(Text, nullable=True)

    # Media agent compatibility fields
    media_query = Column(Text, nullable=True)
    media_url = Column(Text, nullable=True)
    media_selected_source = Column(String(50), nullable=True)
    media_selected_score = Column(String(20), nullable=True)

    region = Column(String(20), default="saudi")  # saudi/global/mixed

    saudi_ratio = Column(Float, default=1.0)  # 0..1 (local vs global bias)

    status = Column(String(20), default="draft")  # draft/active
    media_status = Column(String(30), default="PENDING")  # PENDING/REVIEW_REQUIRED/APPROVED
    media_confidence = Column(Float, default=0.0)
    media_source = Column(String(50), nullable=True)

    current_affairs = Column(Boolean, default=False)

    # Used for dedupe
    signature = Column(String(80), nullable=True, index=True)

    # Hash of normalized stem (EN+AR). Used to prevent repeated questions across runs.
    stem_hash = Column(String(40), nullable=True, index=True)

    # MCQ options (EN + AR)
    option1_en = Column(Text, nullable=True)
    option2_en = Column(Text, nullable=True)
    option3_en = Column(Text, nullable=True)
    option4_en = Column(Text, nullable=True)

    option1_ar = Column(Text, nullable=True)
    option2_ar = Column(Text, nullable=True)
    option3_ar = Column(Text, nullable=True)
    option4_ar = Column(Text, nullable=True)

    correct_option_index = Column(Integer, nullable=True)  # 0..3 (UI uses 0-3; display labels may show 1-4)

    # Final media URL (chosen/suggested)
    url = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    category = relationship("Category", back_populates="questions")
    media_candidates = relationship("MediaCandidate", back_populates="question", cascade="all, delete-orphan")


class MediaCandidate(Base):
    __tablename__ = "media_candidates"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)

    source = Column(String(50), nullable=False)  # wikimedia/unsplash/pexels/youtube/freeaudio/etc
    url = Column(Text, nullable=False)
    title = Column(Text, nullable=True)
    score = Column(Float, default=0.0)
    meta_json = Column(Text, nullable=True)

    selected = Column(Boolean, default=False)  # manual selection flag
    created_at = Column(DateTime, default=datetime.utcnow)

    question = relationship("Question", back_populates="media_candidates")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    entity = Column(String(50), nullable=False)  # category|question|agent
    entity_id = Column(Integer, nullable=True)
    actor = Column(String(80), nullable=True)
    action = Column(String(80), nullable=True)
    message = Column(Text, nullable=False)
    before_json = Column(Text, nullable=True)
    after_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def _sqlite_add_column(conn, table: str, col_def_sql: str) -> None:
    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_def_sql}"))



class LearningEvent(Base):
    __tablename__ = "learning_events"

    id = Column(Integer, primary_key=True, index=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=True)
    actor = Column(String(80), nullable=True)
    event_type = Column(String(80), nullable=False)  # e.g., edit_save, media_select
    before_json = Column(Text, nullable=True)
    after_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def ensure_schema() -> None:
    """
    Create missing tables and add missing columns to existing SQLite DB.
    Keeps UAT/Prod stable without Alembic.
    """
    Base.metadata.create_all(bind=engine)

    if not DATABASE_URL.startswith("sqlite"):
        return

    insp = inspect(engine)
    with engine.begin() as conn:
        # --- categories ---
        if insp.has_table("categories"):
            cols = {c["name"] for c in insp.get_columns("categories")}
            needed = {
                "name_ar": "name_ar VARCHAR(120)",
                "name_en": "name_en VARCHAR(120)",
                "description_ar": "description_ar TEXT",
                "description_en": "description_en TEXT",
                "subtopic": "subtopic VARCHAR(120)",
                "dedup_key": "dedup_key VARCHAR(80)",
                "created_at": "created_at DATETIME",
            }
            for name, ddl in needed.items():
                if name not in cols:
                    _sqlite_add_column(conn, "categories", ddl)

        # --- questions ---
        if insp.has_table("questions"):
            cols = {c["name"] for c in insp.get_columns("questions")}
            needed = {
                "stem_ar": "stem_ar TEXT",
                "stem_en": "stem_en TEXT",
                "answer_ar": "answer_ar TEXT",
                "answer_en": "answer_en TEXT",
                "hint": "hint TEXT",
                "subtopic": "subtopic VARCHAR(120)",
                "difficulty": "difficulty VARCHAR(20) DEFAULT 'medium'",
                "question_type": "question_type VARCHAR(20) DEFAULT 'text'",
                "answer_type": "answer_type VARCHAR(30) DEFAULT 'mcq_selection'",
                "game_type": "game_type VARCHAR(30) DEFAULT \'mcq_selection\'",
                "question_mode": "question_mode VARCHAR(10)",
                "media_intent_json": "media_intent_json TEXT",
                "region": "region VARCHAR(20) DEFAULT 'saudi'",
                "saudi_ratio": "saudi_ratio FLOAT DEFAULT 1.0",
                "status": "status VARCHAR(20) DEFAULT 'draft'",
                "media_status": "media_status VARCHAR(30) DEFAULT 'PENDING'",
                "media_confidence": "media_confidence FLOAT DEFAULT 0.0",
                "media_source": "media_source VARCHAR(50)",
                "media_query": "media_query TEXT",
                "media_url": "media_url TEXT",
                "media_type": "media_type VARCHAR(20)",
                "media_selected_source": "media_selected_source VARCHAR(50)",
                "media_selected_score": "media_selected_score VARCHAR(20)",
                "current_affairs": "current_affairs BOOLEAN DEFAULT 0",
                "signature": "signature VARCHAR(80)",
                "stem_hash": "stem_hash VARCHAR(40)",
                "option1_en": "option1_en TEXT",
                "option2_en": "option2_en TEXT",
                "option3_en": "option3_en TEXT",
                "option4_en": "option4_en TEXT",
                "option1_ar": "option1_ar TEXT",
                "option2_ar": "option2_ar TEXT",
                "option3_ar": "option3_ar TEXT",
                "option4_ar": "option4_ar TEXT",
                "correct_option_index": "correct_option_index INTEGER",
                "url": "url TEXT",
                "created_at": "created_at DATETIME",
                "updated_at": "updated_at DATETIME",
            }
            for name, ddl in needed.items():
                if name not in cols:
                    _sqlite_add_column(conn, "questions", ddl)


        # --- backfills / data fixes ---
        try:
            if insp.has_table("questions"):
                cols_q = {c["name"] for c in insp.get_columns("questions")}
                if "game_type" in cols_q and "answer_type" in cols_q:
                    conn.execute(text("UPDATE questions SET game_type = COALESCE(game_type, answer_type) WHERE game_type IS NULL OR game_type = ''"))
                # Backfill stem_hash for existing rows if column exists
                if "stem_hash" in cols_q:
                    import hashlib, re

                    def _norm(en: str, ar: str) -> str:
                        s = " ".join([(en or ""), (ar or "")]).strip().lower()
                        s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)
                        s = re.sub(r"\s+", " ", s).strip()
                        return s

                    rows = conn.execute(text("SELECT id, stem_en, stem_ar FROM questions WHERE stem_hash IS NULL OR stem_hash = ''")).fetchall()
                    for rid, en, ar in rows:
                        h = hashlib.sha1(_norm(en or "", ar or "").encode("utf-8")).hexdigest()
                        conn.execute(text("UPDATE questions SET stem_hash = :h WHERE id = :id"), {"h": h, "id": rid})
            if insp.has_table("audit_logs"):
                cols_a = {c["name"] for c in insp.get_columns("audit_logs")}
                if "action" in cols_a:
                    conn.execute(text("UPDATE audit_logs SET action = COALESCE(action, message) WHERE action IS NULL OR action = ''"))
        except Exception:
            pass

        # --- media_candidates ---
        if insp.has_table("media_candidates"):
            cols = {c["name"] for c in insp.get_columns("media_candidates")}
            needed = {
                "title": "title TEXT",
                "selected": "selected BOOLEAN DEFAULT 0",
                "created_at": "created_at DATETIME",
                "score": "score FLOAT DEFAULT 0.0",
                "meta_json": "meta_json TEXT",
            }
            for name, ddl in needed.items():
                if name not in cols:
                    _sqlite_add_column(conn, "media_candidates", ddl)

        # --- audit_logs ---
        if insp.has_table("audit_logs"):
            cols = {c["name"] for c in insp.get_columns("audit_logs")}
            needed = {
                "entity": "entity VARCHAR(50)",
                "entity_id": "entity_id INTEGER",
                "actor": "actor VARCHAR(80)",
                "message": "message TEXT",
                "action": "action VARCHAR(80)",
                "before_json": "before_json TEXT",
                "after_json": "after_json TEXT",
                "created_at": "created_at DATETIME",
            }
            for name, ddl in needed.items():
                if name not in cols:
                    _sqlite_add_column(conn, "audit_logs", ddl)


def init_db() -> None:
    ensure_schema()
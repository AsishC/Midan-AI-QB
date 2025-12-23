import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean, inspect, text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./midan.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)

    # Arabic is mandatory/primary; EN is optional
    name_ar = Column(String(120), nullable=False)
    name_en = Column(String(120), nullable=True)

    # Optional metadata / UX fields
    description_ar = Column(Text, nullable=True)
    description_en = Column(Text, nullable=True)
    saudi_safe_notes = Column(Text, nullable=True)  # optional safety/context notes for Saudi content
    scope = Column(String(20), default="saudi")  # saudi/global/mixed
    saudi_ratio = Column(Float, default=1.0)  # 0..1
    subtopic = Column(String(120), nullable=True)
    default_difficulty = Column(String(20), default="medium")  # easy/medium/hard/expert
    sensitivity_level = Column(String(20), default="general")  # general/royal
    dedup_key = Column(String(80), nullable=True, index=True)  # hash for de-dup

    is_current_affairs = Column(Boolean, default=False)
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
    answer_type = Column(String(30), default="mcq_selection")  # mcq_selection/text
    region = Column(String(20), default="saudi")  # saudi/global/mixed

    status = Column(String(20), default="draft")  # draft/active
    media_status = Column(String(30), default="PENDING")  # PENDING/REVIEW_REQUIRED/APPROVED
    media_confidence = Column(Float, default=0.0)
    media_source = Column(String(50), nullable=True)

    current_affairs = Column(Boolean, default=False)

    # Used for dedupe
    signature = Column(String(80), nullable=True, index=True)

    # MCQ options (EN + AR)
    option1_en = Column(Text, nullable=True)
    option2_en = Column(Text, nullable=True)
    option3_en = Column(Text, nullable=True)
    option4_en = Column(Text, nullable=True)

    option1_ar = Column(Text, nullable=True)
    option2_ar = Column(Text, nullable=True)
    option3_ar = Column(Text, nullable=True)
    option4_ar = Column(Text, nullable=True)

    correct_option_index = Column(Integer, nullable=True)  # 1..4

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
    message = Column(Text, nullable=False)
    before_json = Column(Text, nullable=True)
    after_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def _sqlite_add_column(conn, table: str, col_def_sql: str) -> None:
    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_def_sql}"))


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
                "saudi_safe_notes": "saudi_safe_notes TEXT",
                "scope": "scope VARCHAR(20) DEFAULT 'saudi'",
                "saudi_ratio": "saudi_ratio FLOAT DEFAULT 1.0",
                "subtopic": "subtopic VARCHAR(120)",
                "default_difficulty": "default_difficulty VARCHAR(20) DEFAULT 'medium'",
                "sensitivity_level": "sensitivity_level VARCHAR(20) DEFAULT 'general'",
                "dedup_key": "dedup_key VARCHAR(80)",
                "is_current_affairs": "is_current_affairs BOOLEAN DEFAULT 0",
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
                "region": "region VARCHAR(20) DEFAULT 'saudi'",
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
                "before_json": "before_json TEXT",
                "after_json": "after_json TEXT",
                "created_at": "created_at DATETIME",
            }
            for name, ddl in needed.items():
                if name not in cols:
                    _sqlite_add_column(conn, "audit_logs", ddl)


def init_db() -> None:
    ensure_schema()
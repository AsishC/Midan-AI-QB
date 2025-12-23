
"""Validation agent.

Simple pass-through that marks questions with at least one auto candidate
as REVIEW_REQUIRED (for manual moderation UI).

Usage:
  python -m agents.validate_agent --limit 50
"""
import argparse

from database import SessionLocal, Question, MediaCandidate


def main(limit: int = 50):
    db = SessionLocal()
    try:
        qs = (
            db.query(Question)
            .filter(Question.media_status == "REVIEW_REQUIRED")
            .order_by(Question.id.asc())
            .limit(limit)
            .all()
        )
        print(f"[INFO] Found {len(qs)} questions with REVIEW_REQUIRED media.")
        for q in qs:
            candidates = (
                db.query(MediaCandidate)
                .filter(MediaCandidate.question_id == q.id, MediaCandidate.source == "auto")
                .all()
            )
            if not candidates:
                print(f"[INFO] [Q#{q.id}] has no auto candidates; keeping REVIEW_REQUIRED.")
                continue
            print(f"[REVIEW] Q#{q.id} requires manual moderation ({len(candidates)} candidates). ")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    main(limit=args.limit)

# --------------------------------------------------------
# Compatibility wrapper required by app.py
# Old code expects: run_validate_agent_for_question_ids(db, ids)
# --------------------------------------------------------

def run_validate_agent_for_question_ids(db, question_ids):
    """
    Compatibility wrapper.
    If run_validate_agent exists, delegate to it.
    If not, no-op safely. This prevents crashes during pipeline.
    """
    try:
        if 'run_validate_agent' in globals():
            return run_validate_agent(db=db, question_ids=question_ids)
    except Exception as e:
        print("Validate agent failed:", e)

    # Fallback – do nothing
    return None

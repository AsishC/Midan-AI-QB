
Midan Platinum v2 – Question Bank + AI + Media (Free-tier)

How to run
==========

1. Create and activate a virtualenv (Python 3.10+):

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install fastapi uvicorn sqlalchemy openai requests jinja2

3. Set environment variables:

   export OPENAI_API_KEY="sk-..."             # required
   export QB_ADMIN_PASSWORD="midan123"       # optional
   # Optional free image APIs:
   export UNSPLASH_ACCESS_KEY="..."
   export PEXELS_API_KEY="..."

4. Start the app:

   uvicorn app:app --reload

5. Open browser at:

   http://127.0.0.1:8000

   Login with password from QB_ADMIN_PASSWORD.

Agents
======

Media agent (picture/logo only in Free edition):

   python -m agents.media_agent --limit 50

Validation agent (logs questions that need manual moderation):

   python -m agents.validate_agent --limit 50

Batch run (media + validate):

   python -m agents.batch_agent --limit 50

Fact-check helper (flags time-sensitive questions):

   python -m agents.factcheck_agent --limit 100

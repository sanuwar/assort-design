from __future__ import annotations

import json
from datetime import datetime

from sqlmodel import Session

from app.models import Job, JobAttempt


def run_job_pipeline(session: Session, job: Job) -> Job:
    """Scaffold pipeline: create a single mock attempt and mark job completed."""
    job.status = "running"
    session.add(job)
    session.commit()
    session.refresh(job)

    attempt_no = job.attempt_count + 1
    audience = job.audience or job.selected_audience or "cross_functional"

    attempt = JobAttempt(
        job_id=job.id,
        attempt_no=attempt_no,
        audience=audience,
        agent_used="mock",
        generated_summary="Scaffold summary. Pipeline implementation pending.",
        generated_tags_json=json.dumps(["scaffold"]),
        generated_quiz_json=json.dumps(
            ["What is the primary goal of this document?"]
        ),
        evaluator_json=json.dumps(
            {
                "pass": True,
                "word_count": 6,
                "missing_sections": [],
                "fail_reasons": [],
                "fix_instructions": [],
            }
        ),
        passed=True,
        created_at=datetime.utcnow(),
    )

    job.attempt_count = attempt_no
    job.audience = audience
    job.status = "completed"

    session.add(attempt)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job

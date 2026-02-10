# Database Schema – Visual

## Entity Relationship Diagram

```mermaid
erDiagram
    Document {
        int     id          PK
        text    content
        text    source_type
        datetime created_at
    }

    Job {
        int     id                      PK
        int     document_id             FK
        text    selected_audience
        text    audience
        float   routing_confidence
        text    routing_candidates_json
        text    routing_reasons_json
        text    status
        int     attempt_count
        int     max_words
        int     max_retries
        datetime created_at
    }

    JobAttempt {
        int     id                          PK
        int     job_id                      FK
        int     attempt_no
        text    audience
        text    agent_used
        text    generated_one_line_summary
        text    generated_tags_json
        text    generated_clues_json
        text    generated_bullets_json
        text    generated_mindmap
        text    evaluator_json
        bool    passed
        datetime created_at
    }

    Tag {
        int     id      PK
        text    name    "unique, indexed"
    }

    DocumentTag {
        int     document_id     PK,FK
        int     tag_id          PK,FK
    }

    DocumentClue {
        int     id          PK
        int     document_id FK
        text    clue_text
        datetime created_at
    }

    Document  ||--o{  Job         : "has"
    Job       ||--o{  JobAttempt  : "has"
    Document  ||--o{  DocumentClue: "has"
    Document  }o--o{  Tag         : "tagged via"
    DocumentTag }o--|| Document   : ""
    DocumentTag }o--|| Tag        : ""
```

---

## Table Summary

```
Tables & Relationships
│
├── Document                   Core content record
│   ├── id (PK)
│   ├── content
│   ├── source_type
│   └── created_at
│       │
│       ├──< Job                One document → many jobs
│       │     ├── id (PK)
│       │     ├── document_id (FK → Document.id)  [indexed]
│       │     ├── selected_audience
│       │     ├── audience
│       │     ├── routing_confidence
│       │     ├── routing_candidates_json
│       │     ├── routing_reasons_json
│       │     ├── status
│       │     ├── attempt_count
│       │     ├── max_words
│       │     ├── max_retries
│       │     └── created_at
│       │           │
│       │           └──< JobAttempt      One job → many attempts
│       │                 ├── id (PK)
│       │                 ├── job_id (FK → Job.id)  [indexed]
│       │                 ├── attempt_no
│       │                 ├── audience
│       │                 ├── agent_used
│       │                 ├── generated_one_line_summary
│       │                 ├── generated_tags_json
│       │                 ├── generated_clues_json
│       │                 ├── generated_bullets_json
│       │                 ├── generated_mindmap
│       │                 ├── evaluator_json
│       │                 ├── passed
│       │                 └── created_at
│       │
│       ├──< DocumentClue       One document → many clues
│       │     ├── id (PK)
│       │     ├── document_id (FK → Document.id)  [indexed]
│       │     ├── clue_text
│       │     └── created_at
│       │
│       └──< DocumentTag        Join table (many-to-many)
│             ├── document_id (PK, FK → Document.id)  [indexed]
│             └── tag_id (PK, FK → Tag.id)            [indexed]
│
└── Tag                         Global tag registry
      ├── id (PK)
      └── name  [unique, indexed]
```

---

## Indexes

| Index Name                      | Table        | Column(s)   | Notes               |
|---------------------------------|--------------|-------------|---------------------|
| `ix_job_document_id`            | job          | document_id | FK lookup           |
| `ix_jobattempt_job_id`          | jobattempt   | job_id      | FK lookup           |
| `ix_documentclue_document_id`   | documentclue | document_id | FK lookup           |
| `ix_documenttag_document_id`    | documenttag  | document_id | FK lookup           |
| `ix_documenttag_tag_id`         | documenttag  | tag_id      | FK lookup           |
| _(auto)_                        | tag          | name        | unique constraint   |

---

## Job Status Flow

```mermaid
stateDiagram-v2
    [*] --> pending
    pending --> running
    running --> done
    running --> failed
    failed --> running : retry (attempt_count < max_retries)
    failed --> [*]
    done --> [*]
```

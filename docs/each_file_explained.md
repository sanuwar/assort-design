# File Guide - Asort Design

This document summarizes the purpose of each core file and how it fits into the
overall FastAPI + LangGraph application.

## Application Core

### app/main.py
Primary FastAPI entry point. Defines the web routes, loads templates, handles
form submissions, creates documents and jobs, and triggers the pipeline.
This is where the web router and request handlers live.

### app/db.py
Database setup and helpers. Configures the SQLite engine, ensures the database
file exists, and provides a session factory.

### app/models.py
Data models using SQLModel. Defines the schema for documents, jobs, attempts,
tags, and document clues, along with relationships.

### app/graph.py
LangGraph orchestration layer. Currently a scaffold that creates a mock attempt
and marks a job completed; later will run the full agentic pipeline.
This file contains the pipeline "router" step (`route_audience_node`) used to
determine the audience.

### app/llm.py
OpenAI client helper. Detects whether an API key exists and exposes a simple
factory for the OpenAI client, enabling mock mode when missing.

### app/config.py
Configuration loader for agent profiles. Reads `app/agent_profiles.yaml` and
validates the structure for audiences, routing, and evaluation.

### app/agent_profiles.yaml
Prompt and rubric configuration. Defines audience-specific system prompts,
required sections, default word limits, and routing/evaluation prompts.

## Templates

### templates/base.html
Base layout shared by all pages. Provides layout, simple styling, and a
consistent header.

### templates/home.html
Home page UI. Presents input options (paste text, URL, sample content) and
audience selection.

### templates/job_detail.html
Job detail UI. Displays status, audience badge, attempts list, and results like
tags, key clues, decision bullets, and mind map.

### templates/document_detail.html
Document detail UI. Displays document content and any associated tags and
questions.

# Agentic RAG-Based Natural Language to SQL Query Generation and Self-Correction System

An AI-powered Text-to-SQL system that converts natural language questions into SQL queries using Retrieval-Augmented Generation (RAG), agentic query planning, SQL validation, and automatic error correction.

## Overview

This project enables users to interact with relational databases using plain English instead of writing SQL manually. It retrieves relevant database schema information, generates a query plan, constructs SQL using an LLM, validates the query, and attempts to repair errors before execution.

The system supports both **Basic RAG** and **Agentic RAG** modes, providing a framework for comparing a single-call approach with a multi-stage, self-correcting workflow.

## Key Features

- **Natural Language to SQL:** Converts user questions into SQLite queries.
- **Schema-aware RAG:** Retrieves relevant tables, columns, and relationships using embeddings and FAISS.
- **Relationship Expansion:** Identifies intermediate tables required for joins using foreign-key relationships.
- **Agentic Query Planning:** Creates a structured plan containing tables, joins, filters, aggregations, and ordering.
- **Multi-stage SQL Validation:** Checks query safety, syntax, schema references, and database compatibility.
- **Automatic Self-Correction:** Analyzes SQL errors and attempts repairs using an LLM, followed by re-validation.
- **Read-only Execution:** Applies database safety restrictions and a configurable result-row limit.
- **Basic vs. Agentic Modes:** Supports two approaches to natural-language database querying.
- **Execution Trace:** Displays generated SQL, retrieved schema, query plans, validation details, and repair attempts.
- **Offline Testing:** Supports mock LLM execution for testing without external API calls.

## System Workflow

```text
User Question
      |
      v
Schema Retrieval (RAG + FAISS)
      |
      v
Relationship Expansion
      |
      v
Query Planning (LLM)
      |
      v
SQL Generation (LLM)
      |
      v
SQL Validation
      |
      +---- Invalid ----> Error Analysis
      |                        |
      |                        v
      |                   SQL Repair
      |                        |
      |                        +----> Re-validation
      |
      v
Read-only Execution
      |
      v
Result Explanation
```

The repair process is bounded by a configurable maximum number of attempts to prevent endless retries.

## Technology Stack

- **Language:** Python
- **Backend:** FastAPI
- **LLM:** Google Gemini API, with configurable provider support and a mock provider
- **RAG:** FAISS, NumPy, schema chunking, and custom retrieval
- **Database:** SQLite, SQLAlchemy
- **SQL Validation:** SQLGlot and SQLite query-plan validation
- **Frontend:** HTML, CSS, JavaScript
- **Testing:** Pytest

## Demo Database

The project uses a vehicle-rental database containing six relational tables:

- `customers` — customer details
- `vehicles` — vehicle models, categories, and rental prices
- `branches` — branch information
- `rental` — rental transactions and amounts
- `rating` — ratings and reviews
- `maintenance` — vehicle maintenance records

This schema supports questions involving joins, filtering, grouping, aggregations, and ranking.

## SQL Safety

The system incorporates multiple safeguards before query execution:

- Restricts queries to supported read-only `SELECT` and `WITH` statements.
- Rejects destructive SQL operations and multiple statements.
- Validates SQL syntax and table/column references.
- Checks database compatibility using `EXPLAIN QUERY PLAN`.
- Uses read-only database connections and limits returned rows.

These safeguards reduce execution risks but do not guarantee that every generated query is semantically correct.

## Testing and Evaluation Status

**Automated testing:** The project includes 57 offline tests using a mock LLM. These tests cover components such as schema processing, retrieval, SQL validation, database execution, API behavior, and the repair workflow.

**Performance evaluation:** The evaluation framework and dataset are implemented, but the Basic RAG versus Agentic RAG performance evaluation has **not yet been conducted**. No conclusions about comparative accuracy, efficiency, or performance improvements are claimed at this stage.

The planned evaluation metrics include:

- Execution accuracy
- SQL exact match
- Schema retrieval precision and recall
- SQL execution success
- Repair success rate
- Hallucination rate
- Latency and tool usage

Real-LLM evaluation and analysis of the resulting metrics remain future work.

## Example Questions

- How many customers are from Delhi?
- What is the total rental revenue?
- Find the average rating of each vehicle.
- Which vehicles have never been rented?
- Find the top five vehicles by average rating with at least two ratings.
- Calculate rental revenue for each vehicle category.
- Find customers whose total spending exceeds the average customer spending.

## Future Improvements

- Conduct quantitative evaluation of Basic RAG and Agentic RAG.
- Analyze performance across different query difficulty levels.
- Improve schema retrieval and reranking.
- Add PostgreSQL and MySQL support.
- Explore query optimization and feedback-driven SQL correction.
- Add authentication and role-based database access.

---

**Project Goal:** To develop a schema-aware, agentic Text-to-SQL system that combines retrieval, planning, validation, and self-correction to support safer natural-language database querying.
"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You are a helpful assistant that writes SQLite SQL queries.

Given a database schema and an English question, please produce a single SQL query that answers the question correctly.

Rules:
- Use only tables and columns from the provided schema.
- Use SQLite syntax and quote identifiers with double quotes (e.g. "students", "first_name").
- Prefer simple queries over complex ones — if a single SELECT works, don't add a CTE or subquery.
- Use LIMIT for top-N questions (e.g. "top 5", "highest 3").
- Use COUNT(*) for counts.
- Output ONLY the SQL query, wrapped in a ```sql ... ``` fence. No explanation, no extra prose."""


# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Database schema:
{schema}

Question: {question}

Please write the SQLite query."""


VERIFY_SYSTEM = """You are a helpful assistant that checks whether a SQL query's result plausibly answers an English question.

You will be shown the original question, the SQL query that was run, and what the database returned. Your job is to decide whether the returned result actually answers the question.

Common failure modes to watch for:
1. SQL errored — the database returned an error rather than rows.
2. Zero rows when rows are clearly expected (e.g., the question asks for a list or a count of common things).
3. Wrong columns — the returned columns don't match what the question is asking about.
4. Semantic mismatch — the query ran, returned reasonable-looking rows, but answers a different question than the one asked (e.g., missing filter, wrong sort key, wrong aggregation).

Zero rows is sometimes the correct answer (e.g., "list students born in 1850" — there genuinely may be none). Use judgment.

Output ONLY a JSON object, wrapped in a ```json ... ``` fence:

```json
{"ok": true, "issue": ""}
```

or

```json
{"ok": false, "issue": "short description of what's wrong"}
```

Set `ok` to true if the result plausibly answers the question. Set it to false otherwise, and put a brief (one sentence) diagnosis in `issue`. No prose outside the fence."""


# Available placeholders: {question}, {sql}, {execution}
VERIFY_USER = """Question: {question}

SQL that was run:
```sql
{sql}
```

Result:
{execution}

Please decide whether the result answers the question. Respond with the JSON fence as instructed."""


REVISE_SYSTEM = GENERATE_SQL_SYSTEM  # reuse for prefix caching across generate/revise calls


# Available placeholders: {schema}, {question}, {sql}, {execution}, {issue}
REVISE_USER = """Database schema:
{schema}

Question: {question}

A previous attempt at this query was rejected. Here is what happened:

Previous SQL:
```sql
{sql}
```

Result of running it:
{execution}

Why it was rejected: {issue}

Please write a corrected SQLite query that fixes this issue while still answering the original question. Don't change parts of the SQL that weren't called out as wrong."""

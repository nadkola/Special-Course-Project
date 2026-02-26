# Prompt Engineering Design Document

## OLAP Assistant — Tier 2

---

## 1. Overview

This document explains the prompt engineering strategy used to make the LLM reliably translate natural-language business questions into correct, executable pandas code while providing insightful narrative analysis.

The prompt system has three layers:

| Layer              | Purpose                                  | Location         |
|--------------------|------------------------------------------|------------------|
| **System Prompt**  | Persona, schema, rules, output format    | `prompts.py`     |
| **Few-Shot Examples** | Anchor the output format with 5 worked examples | `prompts.py` |
| **User Message**   | Raw question + optional conversation context | `prompts.py` → `build_user_message()` |

---

## 2. Design Principles

### 2.1 Schema-in-Context

The LLM receives the **full dataset schema** (column names, data types, unique values, hierarchies, and key relationships) in every system prompt. This eliminates hallucinated column names and ensures the LLM writes syntactically correct pandas code.

**Why this works:** LLMs generate much more accurate code when they have the exact column names, types, and value ranges available. Without this, the model might guess `region_name` instead of `region`, or assume `quarter` is an integer instead of a string like `'Q4'`.

### 2.2 Structured Output Format

The LLM is instructed to respond with two clearly delimited sections:

```
###ANALYSIS###
<narrative explanation>

###CODE###
<executable pandas code>
```

**Why this works:** By using explicit markers (`###ANALYSIS###` and `###CODE###`), the parsing logic in `parse_llm_response()` can reliably extract both components regardless of how verbose or concise the LLM's response is. This is more robust than asking for JSON or XML, which LLMs sometimes malform.

### 2.3 Constrained Code Environment

The system prompt tells the LLM exactly what it can and cannot do in the code block:

- **Must** assign output to `result`
- **Must** produce a DataFrame
- **Cannot** import anything
- **Cannot** use plotting or Streamlit functions
- **Cannot** use `print()` or `display()`

**Why this works:** These constraints align the LLM's output with the execution sandbox in `app.py`. The app handles visualization separately, so the LLM only needs to produce clean data.

### 2.4 OLAP Operation Vocabulary

The system prompt explicitly lists the six OLAP operations with one-line descriptions and example questions. This gives the LLM a shared vocabulary to:

1. Identify which operation the user is requesting
2. Name the operation in the analysis narrative
3. Choose the right pandas pattern (filter vs. groupby vs. pivot, etc.)

### 2.5 Few-Shot Examples

Five carefully crafted examples cover the core OLAP patterns:

| Example | Operation        | Pattern Demonstrated            |
|---------|------------------|---------------------------------|
| 1       | Group & Summarize | `groupby().agg()` with multiple measures |
| 2       | Slice            | Single filter + aggregation     |
| 3       | Dice             | Multi-filter + country breakdown |
| 4       | Drill-Down       | Time hierarchy navigation       |
| 5       | Compare          | `pivot_table()` with % change calculation |

Each example includes:
- The user's question
- A narrative analysis mentioning the operation name
- Suggested follow-up questions
- Clean, idiomatic pandas code

**Why five examples?** This is enough to establish the pattern without consuming excessive tokens. Each example demonstrates a distinct OLAP operation, so the LLM generalizes to combinations.

---

## 3. Prompt Components in Detail

### 3.1 System Prompt Template (`SYSTEM_PROMPT_TEMPLATE`)

The template has four sections, injected in this order:

1. **Role definition** — "You are an expert Business Intelligence OLAP Assistant"
2. **Schema** — Injected dynamically from `data_utils.get_schema_description()`
3. **OLAP operations table** — All six operations with examples
4. **Response format and rules** — Strict formatting requirements and coding constraints

### 3.2 Schema Injection

The schema is generated at runtime from the actual DataFrame (`get_schema_description()`), meaning:
- If the dataset changes, the prompt updates automatically
- The LLM always has accurate column names, types, and value ranges
- No risk of stale documentation

### 3.3 Few-Shot Examples (`FEW_SHOT_EXAMPLES`)

Appended after the system prompt. Each example uses the exact `###ANALYSIS###` / `###CODE###` format to reinforce the expected output structure.

### 3.4 Conversation Context

The `build_user_message()` function can prepend previous context to the user's current question. The app passes the last 6 messages (3 exchanges) to enable follow-up questions like:

- "Now drill into Q4" (after seeing yearly data)
- "Which country contributed most?" (after seeing regional data)
- "Show that as a trend" (after seeing a comparison)

---

## 4. Error Handling Strategy

### 4.1 Parse Failures

If the LLM response doesn't contain the expected markers, `parse_llm_response()` returns empty strings for the missing sections. The app then displays the raw response as fallback.

### 4.2 Code Execution Errors

If the generated pandas code raises an exception:
1. The error is caught and displayed with a user-friendly message
2. The generated code is still shown (for debugging)
3. A suggestion to rephrase the question is offered
4. The conversation continues normally

### 4.3 Security Filtering

Before execution, the code is scanned for dangerous patterns (`import`, `exec`, `eval`, `open`, `os.`, `sys.`, `subprocess`). If found, execution is blocked with a `SecurityError`.

---

## 5. Design Decisions and Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| English markers over JSON | More robust parsing; LLMs rarely break `###` delimiters | Slightly less structured than JSON |
| Full schema in every call | Maximizes code accuracy | Uses ~300 tokens per call |
| 5 few-shot examples | Good coverage without bloat | Uses ~800 tokens per call |
| Low temperature (0.1) | Analytical consistency | Less creative responses |
| 3-exchange context window | Enables follow-ups without excessive tokens | May lose context in very long sessions |
| Sandbox execution | Prevents malicious code | Cannot use external libraries in generated code |

---

## 6. Iteration and Testing

During development, the prompts were refined through this cycle:

1. **Test a question** → observe the LLM's code output
2. **Identify failures** → wrong column names, bad syntax, missing `result`
3. **Add rules or examples** → address the specific failure mode
4. **Re-test** → confirm the fix doesn't break other queries

Common issues resolved through iteration:
- LLM using `quarter` as integer → added "Quarter values are strings: 'Q1'..." rule
- LLM forgetting `observed=True` → added explicit groupby rule
- LLM generating plots → added "Do NOT use matplotlib/plotly" rule
- LLM not assigning `result` → added few-shot examples all using `result =`

---

## 7. Future Improvements

- **Dynamic few-shot selection:** Choose the most relevant examples based on the user's question type
- **Self-healing code:** If execution fails, automatically send the error back to the LLM for a fix
- **Caching LLM responses:** Cache identical questions to reduce API calls
- **Streaming:** Use streaming API responses for faster perceived performance

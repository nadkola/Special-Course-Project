# 📊 OLAP Assistant — Business Intelligence through Natural Language

A **Tier 2 Streamlit web application** that enables business users to perform OLAP (Online Analytical Processing) operations on a Global Retail Sales dataset using plain-English questions, powered by LLM integration (Anthropic Claude or OpenAI GPT).

---

## Table of Contents

1. [Features](#features)
2. [OLAP Operations Supported](#olap-operations-supported)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Project Structure](#project-structure)
7. [Usage Guide](#usage-guide)
8. [Dataset](#dataset)
9. [Technical Details](#technical-details)

---

## Features

| Feature              | Description                                                  |
|----------------------|--------------------------------------------------------------|
| **Chat Interface**   | Conversational UI — ask questions in plain English            |
| **LLM Integration**  | Supports Anthropic Claude and OpenAI GPT models              |
| **Auto-Visualization** | Automatically picks bar or line charts based on results    |
| **Code Transparency** | View the generated pandas code for every answer              |
| **Quick Queries**    | Pre-built buttons for common analyses                        |
| **Dataset Overview** | Sidebar summary of all dimensions, measures, and hierarchies |
| **Error Handling**   | Graceful recovery from bad code, missing keys, or API errors |
| **Conversation Memory** | Follow-up questions work thanks to chat history context   |
| **Security Sandbox** | Generated code runs in a restricted namespace                |

---

## OLAP Operations Supported

| Operation         | Example Query                              |
|-------------------|--------------------------------------------|
| **Slice**         | "Show only 2024 data"                      |
| **Dice**          | "Electronics in Europe"                    |
| **Group & Summarize** | "Total revenue by region"              |
| **Drill-Down**    | "Break down Q4 2024 by month"              |
| **Roll-Up**       | "Show monthly totals as quarterly"         |
| **Compare**       | "Compare 2023 vs 2024 by region"           |
| **Pivot**         | "Show revenue with regions as columns"     |

---

## Architecture

```
┌────────────────────────────────────────────────┐
│              Streamlit Frontend                 │
│  ┌─────────┐  ┌───────────┐  ┌──────────────┐ │
│  │ Sidebar  │  │ Chat UI   │  │ Metrics Bar  │ │
│  │ Config & │  │ Messages  │  │ KPI Overview │ │
│  │ Overview │  │ + Tables  │  │              │ │
│  │ + Quick  │  │ + Charts  │  │              │ │
│  │ Queries  │  │ + Code    │  │              │ │
│  └─────────┘  └─────┬─────┘  └──────────────┘ │
│                      │                          │
│              ┌───────▼────────┐                 │
│              │  prompts.py    │                 │
│              │  System Prompt │                 │
│              │  + Few-Shot    │                 │
│              │  + Parser      │                 │
│              └───────┬────────┘                 │
│                      │                          │
│         ┌────────────▼──────────────┐           │
│         │     LLM API Call          │           │
│         │  (Anthropic / OpenAI)     │           │
│         └────────────┬──────────────┘           │
│                      │                          │
│         ┌────────────▼──────────────┐           │
│         │  Safe Code Execution      │           │
│         │  Restricted namespace     │           │
│         │  df (pandas DataFrame)    │           │
│         └────────────┬──────────────┘           │
│                      │                          │
│         ┌────────────▼──────────────┐           │
│         │    data_utils.py          │           │
│         │  Cached data loading      │           │
│         │  Schema generation        │           │
│         │  OLAP helper functions    │           │
│         └───────────────────────────┘           │
└────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10 or later
- An API key for **Anthropic** or **OpenAI**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/olap-streamlit-app.git
cd olap-streamlit-app

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the dataset (if not already present)
python generate_dataset.py

# 5. Run the application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Configuration

### API Key Setup (two options)

**Option A — Sidebar (recommended for quick testing):**
Enter your API key directly in the sidebar's "API Key" field. It stays in your browser session and is never stored to disk.

**Option B — Secrets file (recommended for repeated use):**
```bash
# Create the secrets file
mkdir -p .streamlit
nano .streamlit/secrets.toml
```
Add one of:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
# or
OPENAI_API_KEY = "sk-..."
```

> ⚠️ **Never commit `secrets.toml` to version control.** It is already in `.gitignore`.

---

## Project Structure

```
olap-streamlit-app/
├── app.py                  # Main Streamlit application
├── prompts.py              # System prompt, few-shot examples, parser
├── data_utils.py           # Data loading, schema, OLAP helpers
├── generate_dataset.py     # Dataset generator script
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── .streamlit/
│   └── secrets.toml        # API keys (not committed)
├── data/
│   └── global_retail_sales.csv  # 10,000-row retail dataset
├── docs/
│   ├── README.md           # This file
│   └── prompt_design.md    # Prompt engineering documentation
└── screenshots/            # For documentation
```

---

## Usage Guide

### Step 1: Configure the LLM
Select your provider (Anthropic or OpenAI) and enter your API key in the sidebar.

### Step 2: Ask a Question
Type any business question in the chat input at the bottom, or click a **Quick Query** button in the sidebar.

### Step 3: Review Results
Each response includes:
- **Analysis** — a plain-English explanation with the OLAP operation identified
- **Data Table** — the pandas result, expandable
- **Chart** — an auto-generated bar or line chart (when applicable)
- **Generated Code** — the pandas code the LLM wrote (expandable)

### Step 4: Follow Up
Ask follow-up questions — the assistant remembers the conversation context (last 3 exchanges).

### Example Session

```
You:  "Total revenue by region"
Bot:  [Analysis + Table + Bar Chart]

You:  "Drill into North America by country"
Bot:  [Analysis + Table + Bar Chart]

You:  "Compare the top 2 countries year over year"
Bot:  [Analysis + Comparison Table + Chart]
```

---

## Dataset

| Attribute    | Value                                                        |
|-------------|--------------------------------------------------------------|
| Records      | 10,000 transactions                                          |
| Time Period  | January 2023 – December 2025                                 |
| Regions      | North America, Europe, Asia Pacific, Latin America            |
| Categories   | Electronics, Furniture, Office Supplies, Clothing             |
| Segments     | Consumer, Corporate, Home Office                              |
| Countries    | 17                                                            |
| Measures     | quantity, unit_price, revenue, cost, profit, profit_margin    |

### Hierarchies
- **Time:** Year → Quarter → Month
- **Geography:** Region → Country
- **Product:** Category → Subcategory

---

## Technical Details

### LLM Prompt Strategy
See [`docs/prompt_design.md`](docs/prompt_design.md) for the full prompt engineering documentation.

### Security
Generated code is executed in a sandboxed namespace:
- Only `pd`, `np`, and `df` are available
- `import`, `exec`, `eval`, `open`, `os`, `sys` are all blocked
- The DataFrame is passed as a copy to prevent mutation

### Performance
- The CSV is loaded once via `@st.cache_data`
- Schema description is computed once and reused
- Conversation context is limited to the last 3 exchanges to control token usage

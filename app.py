"""
app.py — Business Intelligence OLAP Assistant (Tier 2)
═══════════════════════════════════════════════════════
A Streamlit web application that lets business users perform OLAP analysis
through natural-language conversation, powered by an LLM (Anthropic Claude
or OpenAI GPT).

Architecture:
  User question  →  LLM (with schema context)  →  pandas code + narrative
  pandas code    →  safe execution on cached DataFrame  →  table / chart
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
import re

from data_utils import load_data, get_schema_description, get_dataset_overview
from prompts import build_system_prompt, build_user_message, parse_llm_response


# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="OLAP Assistant — BI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 16px;
    }
    /* Chat messages */
    .stChatMessage { border-radius: 12px; }
    /* Sidebar header */
    [data-testid="stSidebarContent"] h1 { font-size: 1.3rem; }
    /* Result tables */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# LOAD DATA (cached)
# ════════════════════════════════════════════════════════════

@st.cache_data
def cached_load():
    return load_data()

df = cached_load()
schema_text = get_schema_description(df)
overview = get_dataset_overview(df)


# ════════════════════════════════════════════════════════════
# SIDEBAR — Configuration & Dataset Overview
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📊 OLAP Assistant")
    st.caption("Business Intelligence through Natural Language")

    # --- LLM Provider Selection ---
    st.subheader("⚙️ LLM Configuration")
    provider = st.selectbox("Provider", ["Anthropic (Claude)", "OpenAI (GPT)"], index=0)
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Enter your Anthropic or OpenAI API key. It stays in your browser session.",
    )

    # Model selection
    if "Anthropic" in provider:
        model = st.selectbox("Model", [
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20250929",
        ], index=0)
    else:
        model = st.selectbox("Model", [
            "gpt-4o",
            "gpt-4o-mini",
        ], index=0)

    st.divider()

    # --- Dataset Overview ---
    st.subheader("📋 Dataset Overview")
    st.markdown(f"**Records:** {overview['total_rows']:,}")
    st.markdown(f"**Period:** {overview['date_range']}")
    st.markdown(f"**Revenue:** ${overview['total_revenue']:,.2f}")
    st.markdown(f"**Profit:** ${overview['total_profit']:,.2f}")
    st.markdown(f"**Avg Margin:** {overview['avg_profit_margin']:.1f}%")

    with st.expander("Dimensions"):
        st.markdown(f"**Regions ({len(overview['regions'])}):** {', '.join(overview['regions'])}")
        st.markdown(f"**Categories ({len(overview['categories'])}):** {', '.join(overview['categories'])}")
        st.markdown(f"**Segments ({len(overview['segments'])}):** {', '.join(overview['segments'])}")
        st.markdown(f"**Countries:** {overview['countries_count']}")
        st.markdown(f"**Years:** {', '.join(str(y) for y in overview['years'])}")

    with st.expander("Hierarchies"):
        st.markdown("🕐 **Time:** Year → Quarter → Month")
        st.markdown("🌍 **Geography:** Region → Country")
        st.markdown("📦 **Product:** Category → Subcategory")
        st.markdown("👤 **Customer:** Segment")

    st.divider()

    # --- Quick Actions ---
    st.subheader("⚡ Quick Queries")
    quick_queries = {
        "Revenue by Region": "Show total revenue by region",
        "2024 by Quarter": "Break down 2024 revenue by quarter",
        "Electronics in Europe": "Show Electronics sales in Europe by country",
        "2023 vs 2024": "Compare 2023 vs 2024 total revenue by region",
        "Top 5 Countries": "What are the top 5 countries by profit?",
        "Monthly Trend 2024": "Show monthly revenue trend for 2024",
        "Category Margins": "Which category has the highest profit margin?",
        "Worst Subcategory": "Identify the worst-performing subcategory by profit",
    }
    for label, query in quick_queries.items():
        if st.button(f"▶ {label}", use_container_width=True, key=f"quick_{label}"):
            st.session_state["pending_query"] = query

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ════════════════════════════════════════════════════════════
# LLM INTEGRATION
# ════════════════════════════════════════════════════════════

def call_llm(user_query: str, conversation_history: list) -> str:
    """
    Send the user's question to the configured LLM and return the raw response.
    Supports Anthropic Claude and OpenAI GPT.
    """
    system_prompt = build_system_prompt(schema_text)

    if "Anthropic" in provider:
        try:
            from anthropic import Anthropic
        except ImportError:
            st.error("Please install the Anthropic SDK: `pip install anthropic`")
            return ""

        client = Anthropic(api_key=api_key)

        # Build messages from conversation history
        messages = []
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    else:  # OpenAI
        try:
            from openai import OpenAI
        except ImportError:
            st.error("Please install the OpenAI SDK: `pip install openai`")
            return ""

        client = OpenAI(api_key=api_key)

        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=0.1,  # Low temperature for analytical consistency
        )
        return response.choices[0].message.content


# ════════════════════════════════════════════════════════════
# SAFE CODE EXECUTION
# ════════════════════════════════════════════════════════════

def execute_pandas_code(code: str, dataframe: pd.DataFrame) -> pd.DataFrame | None:
    """
    Safely execute LLM-generated pandas code against the DataFrame.

    Security measures:
      • Code runs in a restricted namespace (only pd, np, df).
      • Dangerous builtins (exec, eval, open, __import__) are blocked.
      • Output must be a DataFrame assigned to `result`.
    """
    # Basic security checks
    dangerous_patterns = [
        r"\bimport\b", r"\bexec\b", r"\beval\b", r"\bopen\b",
        r"\b__import__\b", r"\bos\.", r"\bsys\.", r"\bsubprocess\b",
        r"\bshutil\b", r"\bpathlib\b",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            # Allow 'import' only if it's inside a comment or part of pandas syntax
            if pattern == r"\bimport\b":
                # Remove comments and string literals, then check
                cleaned = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
                cleaned = re.sub(r"'[^']*'|\"[^\"]*\"", "", cleaned)
                if re.search(pattern, cleaned):
                    raise SecurityError(f"Blocked potentially unsafe operation: {pattern}")
            else:
                raise SecurityError(f"Blocked potentially unsafe operation: {pattern}")

    # Prepare execution namespace
    namespace = {
        "df": dataframe.copy(),
        "pd": pd,
        "np": np,
        "result": None,
    }

    exec(code, {"__builtins__": {}}, namespace)

    result = namespace.get("result")
    if result is None:
        raise ValueError("Code did not produce a `result` variable.")
    if not isinstance(result, pd.DataFrame):
        # Try to convert Series to DataFrame
        if isinstance(result, pd.Series):
            result = result.to_frame().reset_index()
        else:
            raise TypeError(f"Expected DataFrame, got {type(result).__name__}")
    return result


class SecurityError(Exception):
    """Raised when LLM-generated code contains unsafe operations."""
    pass


# ════════════════════════════════════════════════════════════
# AUTO-CHART LOGIC
# ════════════════════════════════════════════════════════════

def auto_chart(result_df: pd.DataFrame):
    """
    Automatically select and render an appropriate chart based on the
    shape of the result DataFrame.
    """
    if result_df is None or result_df.empty or len(result_df) < 2:
        return

    # Identify numeric and text columns
    num_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = result_df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols or not text_cols:
        return

    label_col = text_cols[0]
    # Pick the most meaningful numeric column (prefer 'revenue', 'profit', etc.)
    priority = ["revenue", "profit", "quantity", "orders", "profit_margin"]
    value_col = None
    for p in priority:
        matches = [c for c in num_cols if p in c.lower()]
        if matches:
            value_col = matches[0]
            break
    if value_col is None:
        value_col = num_cols[0]

    chart_df = result_df[[label_col, value_col]].copy()
    chart_df = chart_df.set_index(label_col)

    # Decide chart type
    n_rows = len(chart_df)
    is_time_series = label_col.lower() in [
        "month", "month_name", "quarter", "year", "order_date",
    ]

    if is_time_series and n_rows <= 24:
        st.line_chart(chart_df, use_container_width=True)
    elif n_rows <= 15:
        st.bar_chart(chart_df, use_container_width=True)
    # Skip chart for very large result sets


# ════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ════════════════════════════════════════════════════════════

st.title("📊 Business Intelligence OLAP Assistant")
st.markdown(
    "Ask questions about the **Global Retail Sales** dataset in plain English. "
    "The assistant translates your questions into OLAP operations and returns "
    "formatted results with insights."
)

# --- Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Display Overview Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${overview['total_revenue']:,.0f}")
col2.metric("Total Profit", f"${overview['total_profit']:,.0f}")
col3.metric("Avg Margin", f"{overview['avg_profit_margin']:.1f}%")
col4.metric("Transactions", f"{overview['total_rows']:,}")

st.divider()

# --- Chat History ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Re-render assistant messages with their saved components
            if "analysis" in msg:
                st.markdown(msg["analysis"])
            if "result_df" in msg and msg["result_df"] is not None:
                with st.expander("📋 Data Table", expanded=True):
                    st.dataframe(msg["result_df"], use_container_width=True)
                auto_chart(msg["result_df"])
            if "code" in msg and msg["code"]:
                with st.expander("🔧 Generated Code"):
                    st.code(msg["code"], language="python")
            if "error" in msg:
                st.error(msg["error"])
        else:
            st.markdown(msg["content"])

# --- Handle pending quick queries ---
pending = st.session_state.pop("pending_query", None)

# --- Chat Input ---
user_input = st.chat_input("Ask a question about the data…")
query = pending or user_input

if query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state["messages"].append({"role": "user", "content": query})

    # Validate API key
    if not api_key:
        with st.chat_message("assistant"):
            err = "⚠️ Please enter your API key in the sidebar to get started."
            st.warning(err)
        st.session_state["messages"].append({
            "role": "assistant", "content": err, "analysis": err,
        })
    else:
        # Call LLM
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question…"):
                try:
                    # Build conversation for context
                    conv_for_llm = []
                    # Include last 6 messages for context (3 exchanges)
                    recent = st.session_state["messages"][-7:]  # includes current user msg
                    for m in recent:
                        conv_for_llm.append({
                            "role": m["role"],
                            "content": m["content"],
                        })

                    raw_response = call_llm(query, conv_for_llm)
                    parsed = parse_llm_response(raw_response)

                    analysis = parsed["analysis"]
                    code = parsed["code"]
                    result_df = None
                    error_msg = None

                    # Display analysis
                    if analysis:
                        st.markdown(analysis)

                    # Execute code
                    if code:
                        try:
                            result_df = execute_pandas_code(code, df)

                            # Display results
                            if result_df is not None and not result_df.empty:
                                with st.expander("📋 Data Table", expanded=True):
                                    st.dataframe(
                                        result_df,
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                auto_chart(result_df)
                            else:
                                st.info("The query returned no results. Try adjusting your filters.")

                        except SecurityError as se:
                            error_msg = f"🔒 Security: {se}"
                            st.error(error_msg)
                        except Exception as exec_err:
                            error_msg = f"⚠️ Code execution error: {exec_err}"
                            st.error(error_msg)
                            st.caption("The LLM-generated code encountered an error. Try rephrasing your question.")

                        # Always show the generated code
                        with st.expander("🔧 Generated Code"):
                            st.code(code, language="python")

                    # Save assistant message
                    assistant_msg = {
                        "role": "assistant",
                        "content": analysis or raw_response,
                        "analysis": analysis,
                        "code": code,
                        "result_df": result_df if result_df is not None else None,
                    }
                    if error_msg:
                        assistant_msg["error"] = error_msg
                    st.session_state["messages"].append(assistant_msg)

                except Exception as e:
                    error_text = f"❌ Error communicating with the LLM: {str(e)}"
                    st.error(error_text)
                    st.caption(
                        "Check that your API key is correct and the selected model is available."
                    )
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": error_text,
                        "analysis": error_text,
                        "error": str(e),
                    })


# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "💡 **Tip:** Try questions like *'Total revenue by region'*, "
    "*'Compare 2023 vs 2024'*, or *'Drill down Q4 2024 by month'*. "
    "Use the Quick Queries in the sidebar to get started."
)

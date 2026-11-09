"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb

Pipeline (logic, prompts and SQL are unchanged from the notebook):
    classify_intent -> (verified template | generate_query) -> validate_query
    -> (retry_generation once, generated track only) -> execute_query
    -> generate_response -> escalate to a human analyst if validation still fails.

Run locally:
    streamlit run app.py

Required files next to app.py:
    credit_risk_portfolio.db   (SQLite database, opened READ-ONLY)
    test_queries2.csv          (ground-truth test cases; optional, enables the
                                example questions and the Evaluation tab)

Required secret / environment variable:
    OPENAI_API_KEY             (optional: OPENAI_BASE_URL, OPENAI_MODEL, DB_PATH,
                                TEST_QUERIES_PATH)
"""

import json
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI


# =============================================================================
# 1. CONFIGURATION (Streamlit secrets -> environment variables -> config.json)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent


def _load_config_json():
    """Optional fallback: the same config.json format used in the notebook."""
    for candidate in (BASE_DIR / "config.json", Path.cwd() / "config.json"):
        if candidate.exists():
            try:
                with open(candidate, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


def get_setting(name, default=None):
    """Look a setting up in st.secrets, then os.environ, then config.json."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass  # no secrets.toml available
    if os.environ.get(name):
        return os.environ[name]
    cfg = _load_config_json()
    if cfg.get(name):
        return str(cfg[name])
    return default


def get_openai_credentials():
    api_key = get_setting("OPENAI_API_KEY")
    # Same fallback as the notebook: OPENAI_API_BASE or OPENAI_BASE_URL
    base_url = get_setting("OPENAI_BASE_URL") or get_setting("OPENAI_API_BASE")
    return api_key, base_url


MODEL_NAME = get_setting("OPENAI_MODEL", "gpt-4o-mini")
DB_PATH = get_setting("DB_PATH", "credit_risk_portfolio.db")
TEST_QUERIES_PATH = get_setting("TEST_QUERIES_PATH", "test_queries2.csv")


def resolve_path(p):
    """Resolve a (possibly relative) path against the app folder, then the CWD."""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    for root in (BASE_DIR, Path.cwd()):
        if (root / path).exists():
            return (root / path).resolve()
    return (BASE_DIR / path).resolve()


def find_test_queries_file():
    names = [TEST_QUERIES_PATH] + [
        n for n in ("test_queries2.csv", "test_queries.csv") if n != TEST_QUERIES_PATH
    ]
    for name in names:
        candidate = resolve_path(name)
        if candidate.exists():
            return candidate
    return None


# =============================================================================
# 2. LLM SETUP (same model and temperature as the notebook; two instances)
# =============================================================================
@st.cache_resource(show_spinner=False)
def _build_llm(model_name, api_key, base_url, role):
    """role ('llm' / 'evaluator_llm') keeps two separate instances, as in the notebook."""
    os.environ["OPENAI_API_KEY"] = str(api_key)
    if base_url:
        os.environ["OPENAI_BASE_URL"] = str(base_url)
    kwargs = {"model": model_name, "temperature": 0, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _require_credentials():
    api_key, base_url = get_openai_credentials()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to .streamlit/secrets.toml, "
            "set it as an environment variable, or provide a config.json."
        )
    return api_key, base_url


def get_llm():
    api_key, base_url = _require_credentials()
    return _build_llm(MODEL_NAME, api_key, base_url, "llm")


def get_evaluator_llm():
    api_key, base_url = _require_credentials()
    return _build_llm(MODEL_NAME, api_key, base_url, "evaluator_llm")


# =============================================================================
# 3. DATABASE + GROUND TRUTH LOADING
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_connection(db_file_str):
    """READ-ONLY SQLite connection (URI mode with ?mode=ro), as in the notebook."""
    uri = f"{Path(db_file_str).as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


@st.cache_data(show_spinner=False)
def load_ground_truth(path_str):
    return pd.read_csv(path_str)


# Database schema provided to the LLM in the prompts
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# =============================================================================
# 4. VERIFIED QUERY TEMPLATE LIBRARY (SQL kept exactly as in the notebook)
# =============================================================================
# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """
SELECT
    sm.sector_name,
    COUNT(lm.loan_account_number) AS loan_count,
    ROUND(SUM(lm.total_outstanding) / 1e6, 2) AS total_outstanding,
    ROUND(SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                    THEN lm.total_outstanding ELSE 0 END) / 1e6, 2) AS npa_outstanding_m
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_outstanding DESC;
"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """
SELECT
    loan_category,
    COUNT(loan_account_number) AS loan_count,
    SUM(total_outstanding) / 1000000.0 AS total_outstanding_in_millions
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_in_millions DESC;
"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(loan_account_number) AS loan_count,
    SUM(ead_amount) / 1000000.0 AS total_ead,
    SUM(ecl_amount) / 1000000.0 AS total_ecl
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage;
"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """
SELECT
    s.sector_name,
    AVG(p.provision_coverage_ratio) AS avg_provision_coverage_ratio
FROM provisioning p
INNER JOIN loan_master l ON p.loan_account_number = l.loan_account_number
INNER JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC;
"""

# VQ5: Top 10 Loan Exposures
sql_5 = """
SELECT
    l.borrower_name,
    s.sector_name,
    l.total_outstanding / 1000000.0 AS outstanding_amount_in_millions,
    l.asset_classification
FROM loan_master l
INNER JOIN sector_master s ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10;
"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """
SELECT
    group_name,
    COUNT(loan_account_number) AS loan_count,
    SUM(total_outstanding) / 1000000.0 AS total_outstanding_in_millions
FROM loan_master
WHERE group_name IS NOT NULL AND group_name != ''
GROUP BY group_name
ORDER BY total_outstanding_in_millions DESC
LIMIT 5;
"""

# VQ7: All Overdue Loan Accounts
sql_7 = """
SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    l.total_outstanding / 1000000.0 AS total_outstanding_in_millions,
    l.days_past_due,
    l.asset_classification
FROM loan_master l
INNER JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC;
"""

# VQ8: DPD Bucket Distribution
sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(loan_account_number) AS loan_count,
    SUM(total_outstanding) / 1000000.0 AS total_outstanding_in_millions
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE
        WHEN dpd_bucket = '0 (Current)' THEN 1
        WHEN dpd_bucket = '1-30' THEN 2
        WHEN dpd_bucket = '31-60' THEN 3
        WHEN dpd_bucket = '61-90' THEN 4
        ELSE 5
    END;
"""

# VQ9: Latest Rating Downgrades
sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30' AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;
"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """
SELECT
    reporting_date,
    SUM(ecl_amount) / 1000000.0 AS total_ecl_in_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC;
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },
    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },
    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },
    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },
    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },
    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },
    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },
    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },
    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },
    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# =============================================================================
# 5. TOOL DEFINITIONS (notebook code; only the LLM handles are swapped for the
#    cached get_llm() / get_evaluator_llm() accessors)
# =============================================================================
def _strip_trailing_semicolons(sql):
    """Port helper: drop surrounding whitespace and any trailing ';' from a SQL string."""
    return sql.strip().rstrip(';').strip()


# ---- Intent Classification Tool ----
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
### ROLE
You are a query router for a financial operations analytics system. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording.
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about shape of answer: a question asking for row-level detail should NOT match a template that returns an aggregate count.


### OUTPUT
Return ONLY a valid, parseable JSON dictionary with these exact keys. Do not include markdown wraps or any conversational filler:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
"""

    response = get_llm().invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ---- Query Generation Tool ----
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
System: You are an expert SQL engineer. Generate a single, valid, read-only SQLite SELECT query to answer the user's question.
Do not write any introductory or explanatory text. Return ONLY the raw SQL statement.

Database Schema:
{schema_context}

User Question:
{user_question}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. Ensure the query is SQLite compatible.
5. For monetary amounts: use ROUND(... / 1e6, 2) to convert to millions.
6. NPA defenition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
7. For the latest provisioning data, use: reporting_date = '2025-09-30'
8. For the latest rating data, use: rating_date = '2025-09-30'
9. In SQLite, never subtract DATE() or date columns directly (e.g. DATE(a)-DATE(b)) — it silently returns 0; always use julianday(a)-julianday(b) for day differences.
10. Alias every numeric column with a suffix that states its unit, so the result is self-describing. Use _usd for dollar amounts, _pct or _percent for percentages, _count for counts, _hours for hour values, and _days for day values. Avoid bare aliases like "value", "amount", or "total".

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.
"""

    sql = get_llm().invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ---- Query Validation Tool ----
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    # PORT FIX: .strip() added so the trailing ';' + newline of the stored verified templates
    # is not mistaken for a second statement (internal ';' is still rejected).
    if ';' in candidate_sql.strip().rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
System: You are an expert database auditor. Your job is to check if the candidate SQL query is relevant and correct to answer the user's original question.

Context:
{track_context}

User Question:
{user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.

### OUTPUT
Return ONLY a JSON dictionary with these exact keys:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence explaining the verdict"
}}
"""
    relevance_response = get_evaluator_llm().invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            # PORT FIX: probe the output shape through a sub-select; appending LIMIT 0 breaks on the
            # templates' trailing ';' and on templates that already end in LIMIT (VQ5, VQ6).
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({_strip_trailing_semicolons(expected_sql)}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({_strip_trailing_semicolons(candidate_sql)}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ---- Retry Generation Tool ----
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.
    '''

    retry_prompt = f"""
System: You are an expert SQL engineer. Your previous SQL failed validation with an error. Please correct the query based on the feedback.
Do not write any introductory or explanatory text. Return ONLY the corrected, valid, read-only SQLite SELECT statement.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.
"""

    revised_sql = get_llm().invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ---- Query Execution Tool ----
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ---- Response Generation Tool ----
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.
    '''

    response_prompt = f"""
System: You are a credit risk portfolio manager. Summarize the analytical results in the pandas DataFrame below to directly and professionally answer the user's question.
Be concise, precise, and highlight key metrics and figures.

User Question:
{user_question}

Query Results (DataFrame):
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the user's specific question directly. Do not dump the entire table.
2. Provide context from other rows only when it adds value (for example, ranking or comparison).
3. State exact numbers from the data. Do not round beyond what is shown.
4. Flag anything notable.
5. Use clear, professional language suitable for a memo.
6. Keep the response focused. Two to four sentences for simple questions, up to a short paragraph for complex ones.

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.
""
"""

    narrative = get_llm().invoke(response_prompt).content.strip()
    return narrative


# =============================================================================
# 6. PIPELINE ORCHESTRATION
#    Identical to the notebook, except the verbose print() output is collected
#    into the `trace` list so the UI can show it instead of a console.
# =============================================================================
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True, trace=None):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records intermediate pipeline stages.
    - trace (list): Optional list that receives the intermediate stage messages.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    def _say(message):
        if verbose and trace is not None:
            trace.append(message)

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    if verbose:
        _say(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
        _say(f"    Reason: {log['match_reason']}")


    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    if verbose:
        _say(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")


    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    if verbose:
        _say(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            _say(f"    Failed check: {gate.get('failed_check')}")
            _say(f"    Details: {gate.get('details')}")


    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        if verbose:
            _say(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        if verbose:
            _say(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
            if not gate['passed']:
                _say(f"    Retry failed check: {gate.get('failed_check')}")
                _say(f"    Retry details: {gate.get('details')}")


    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        if verbose:
            _say(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}


    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    if verbose:
        _say(f"[4] Execute: {len(df)} rows returned")
        if exec_result['warnings']:
            _say(f"    Warnings: {exec_result['warnings']}")


    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative


    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    if verbose:
        _say(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# =============================================================================
# 7. EVALUATION HELPERS (notebook "Evaluation Against Ground Truth" section)
# =============================================================================
NOTEBOOK_EXPECTED_ROUTES = ['verified', 'verified', 'generated', 'generated', 'generated']
NOTEBOOK_EXPECTED_QUERY_IDS = ['VQ1', 'VQ8', None, None, None]
NUM_TEST_CASES = 5


def get_test_question(ground_truth, i):
    """Same lookup as the notebook test cells."""
    if 'question' in ground_truth.columns:
        return ground_truth['question'].iloc[i]
    return ground_truth['User Query'].iloc[i]


def get_expected(ground_truth, i):
    """Expected route / query id: the CSV columns if present, else the notebook values."""
    route = NOTEBOOK_EXPECTED_ROUTES[i] if i < len(NOTEBOOK_EXPECTED_ROUTES) else None
    qid = NOTEBOOK_EXPECTED_QUERY_IDS[i] if i < len(NOTEBOOK_EXPECTED_QUERY_IDS) else None
    if ground_truth is not None:
        if 'Expected Route' in ground_truth.columns and pd.notna(ground_truth['Expected Route'].iloc[i]):
            route = str(ground_truth['Expected Route'].iloc[i]).strip().lower()
        if 'Expected Query ID' in ground_truth.columns:
            value = ground_truth['Expected Query ID'].iloc[i]
            qid = None if pd.isna(value) else str(value).strip()
    return route, qid


def build_evaluation(test_results, expected):
    evaluation_rows = []

    for i, tr in enumerate(test_results):
        expected_route, expected_query_id = expected[i]

        actual_route = tr.get('route')
        actual_query_id = tr.get('query_id')
        confidence = tr.get('confidence')
        row_count = tr.get('row_count')

        evaluation_rows.append({
            'Test Case': f'TC{i+1:03d}',
            'Expected Route': expected_route,
            'Actual Route': actual_route,
            'Route Match': actual_route == expected_route,
            'Expected Query ID': expected_query_id,
            'Actual Query ID': actual_query_id,
            'Query ID Match': (expected_query_id is None and actual_query_id is None) or (actual_query_id == expected_query_id),
            'Confidence': confidence,
            'Rows Returned': row_count
        })

    evaluation_df = pd.DataFrame(evaluation_rows)

    path_accuracy = evaluation_df['Route Match'].mean() * 100

    # Calculate query accuracy on verified expected routes
    verified_mask = evaluation_df['Expected Route'] == 'verified'
    query_accuracy = evaluation_df.loc[verified_mask, 'Query ID Match'].mean() * 100

    # Filter out any non-numeric or None confidence scores (like 'ESCALATED') to compute mean
    numeric_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce')
    average_confidence = numeric_confidence.mean()

    return evaluation_df, path_accuracy, query_accuracy, average_confidence


# =============================================================================
# 8. STREAMLIT UI HELPERS
# =============================================================================
EXAMPLE_PLACEHOLDER = "-- Choose an example question --"
SANITY_CHECK_QUESTION = "Show me the sector-wise outstanding and NPA breakdown"


def md_safe(text):
    """Escape '$' so dollar amounts are not rendered as LaTeX by Streamlit markdown."""
    return str(text).replace("$", "\\$")


def fmt_confidence(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.2f}"
    return str(value) if value is not None else "n/a"


def run_one_question(question, conn):
    """Run the full pipeline and capture the verbose trace."""
    trace = []
    result = run_pipeline(
        question, conn, verified_query_library, database_schema,
        verbose=True, trace=trace,
    )
    result["trace"] = trace
    return result


def render_result(result, show_trace=True, key_prefix="result"):
    log = result["log"]
    df = result.get("dataframe")

    if result.get("escalated"):
        st.warning("Escalated to a human analyst. The query could not be reliably resolved.")
        st.markdown(md_safe(result.get("narrative") or ""))
        gate = log.get("gate_result") or {}
        st.caption(f"Failed check: {gate.get('failed_check')}  |  Route: {log.get('route')}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", str(log.get("route") or "n/a").title())
        c2.metric("Query ID", log.get("query_id") or "-")
        c3.metric("Confidence", fmt_confidence(log.get("confidence")))
        c4.metric("Rows returned", log.get("row_count") if log.get("row_count") is not None else "-")

        st.markdown("#### Answer")
        st.info(md_safe(result.get("narrative") or ""))

    if log.get("match_reason"):
        st.caption(f"Router reasoning: {md_safe(log['match_reason'])}")
    if log.get("retry_used"):
        st.caption("SQL was regenerated once after a validation failure.")

    sql_to_show = log.get("executed_sql") or log.get("candidate_sql")
    if sql_to_show:
        label = "Executed SQL" if log.get("executed_sql") else "Candidate SQL (not executed)"
        with st.expander(label, expanded=not result.get("escalated")):
            st.code(sql_to_show, language="sql")

    if df is not None:
        st.markdown("#### Result data")
        st.dataframe(df)
        st.download_button(
            "Download results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="query_results.csv",
            mime="text/csv",
            key=f"{key_prefix}_download",
        )

    with st.expander("Validation gate details"):
        st.json(log.get("gate_result") or {})

    if show_trace and result.get("trace"):
        with st.expander("Pipeline trace"):
            st.code("\n".join(result["trace"]), language="text")


def init_state():
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("eval_results", None)
    st.session_state.setdefault("question_input", "")


def _use_example():
    choice = st.session_state.get("example_choice")
    if choice and choice != EXAMPLE_PLACEHOLDER:
        st.session_state["question_input"] = choice


def _clear_question():
    st.session_state["question_input"] = ""
    st.session_state["example_choice"] = EXAMPLE_PLACEHOLDER
    st.session_state["last_result"] = None
    st.session_state["last_error"] = None


# =============================================================================
# 9. TABS
# =============================================================================
def tab_ask(conn, ground_truth, show_trace):
    st.markdown(
        "Ask a business question about the commercial lending portfolio in plain English. "
        "Recurring questions are answered from the **Verified Query Library**; novel questions "
        "get freshly generated SQL that must pass the validation gate before it runs."
    )

    examples = []
    if ground_truth is not None and 'User Query' in ground_truth.columns:
        examples.extend(ground_truth['User Query'].dropna().astype(str).tolist())
    if SANITY_CHECK_QUESTION not in examples:
        examples.append(SANITY_CHECK_QUESTION)

    st.selectbox(
        "Example questions",
        [EXAMPLE_PLACEHOLDER] + examples,
        key="example_choice",
        on_change=_use_example,
    )
    st.text_area(
        "Your question",
        key="question_input",
        height=110,
        placeholder="e.g. Show me the sector-wise outstanding and NPA breakdown",
    )

    col_run, col_clear, _ = st.columns([1, 1, 6])
    run_clicked = col_run.button("Run query", type="primary")
    col_clear.button("Clear", on_click=_clear_question)

    if run_clicked:
        question = st.session_state["question_input"].strip()
        if not question:
            st.warning("Please enter a question first.")
        else:
            st.session_state["last_error"] = None
            with st.spinner("Running pipeline: classify, build, validate, execute, summarise..."):
                try:
                    result = run_one_question(question, conn)
                    st.session_state["last_result"] = result
                    st.session_state["history"].append({
                        "Question": question,
                        "Route": result.get("route"),
                        "Query ID": result.get("query_id") or "-",
                        "Confidence": fmt_confidence(result.get("confidence")),
                        "Rows": result.get("row_count") if result.get("row_count") is not None else "-",
                    })
                except Exception as exc:  # LLM/API/parse errors should not crash the app
                    st.session_state["last_result"] = None
                    st.session_state["last_error"] = f"{type(exc).__name__}: {exc}"

    if st.session_state["last_error"]:
        st.error(f"The pipeline failed unexpectedly. {st.session_state['last_error']}")

    if st.session_state["last_result"] is not None:
        st.divider()
        render_result(st.session_state["last_result"], show_trace, key_prefix="ask")


def tab_library(conn):
    st.markdown(
        "The Verified Query Template Library holds ten pre-approved SQL queries. Each entry has a "
        "unique ID, a plain-English description used for intent matching, and a fixed SQL statement "
        "that runs without modification."
    )
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry['sql'].strip(), language="sql")
            if st.button(f"Run {qid} directly", key=f"run_{qid}"):
                try:
                    st.dataframe(pd.read_sql_query(entry['sql'], conn))
                except Exception as exc:
                    st.error(f"{type(exc).__name__}: {exc}")


def tab_schema(conn):
    st.markdown("Schema context passed to the LLM in the SQL generation, retry and validation prompts.")
    with st.expander("Database schema", expanded=False):
        st.code(database_schema.strip(), language="text")

    st.markdown("#### Table previews (first 3 rows)")
    for table_name in ['sector_master', 'loan_master', 'borrower_rating', 'provisioning']:
        with st.expander(table_name):
            try:
                st.dataframe(pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn))
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")


def tab_evaluation(conn, ground_truth):
    st.markdown(
        "Runs the five ground-truth test cases end-to-end and compares the selected route and "
        "query template with the expected values. Reports **Selected Path Accuracy**, "
        "**Selected Query Accuracy** and **Average Confidence Score**."
    )
    if ground_truth is None:
        st.info(
            "No ground-truth CSV found. Place `test_queries2.csv` next to `app.py` "
            "(or set TEST_QUERIES_PATH) to enable this tab."
        )
        return

    st.dataframe(ground_truth)
    n_cases = min(NUM_TEST_CASES, len(ground_truth))

    if st.button(f"Run all {n_cases} test cases", type="primary", key="run_eval"):
        results = []
        progress = st.progress(0.0, text="Starting...")
        for i in range(n_cases):
            question = get_test_question(ground_truth, i)
            progress.progress(i / n_cases, text=f"Test case {i + 1}/{n_cases}: {question[:80]}")
            try:
                results.append(run_one_question(question, conn))
            except Exception as exc:
                results.append({
                    'route': None, 'query_id': None, 'confidence': None, 'row_count': None,
                    'narrative': f"Pipeline error: {type(exc).__name__}: {exc}",
                    'executed_sql': None, 'dataframe': None, 'trace': [],
                    'escalated': True, 'log': {'gate_result': {}},
                })
        progress.progress(1.0, text="Done")
        st.session_state["eval_results"] = results

    results = st.session_state.get("eval_results")
    if not results:
        return

    expected = [get_expected(ground_truth, i) for i in range(len(results))]
    evaluation_df, path_acc, query_acc, avg_conf = build_evaluation(results, expected)

    m1, m2, m3 = st.columns(3)
    m1.metric("Selected Path Accuracy", f"{path_acc:.1f}%")
    m2.metric("Selected Query Accuracy", "n/a" if pd.isna(query_acc) else f"{query_acc:.1f}%")
    m3.metric("Average Confidence Score", "n/a" if pd.isna(avg_conf) else f"{avg_conf:.2f}")

    display_df = evaluation_df.copy()
    for col in ('Confidence', 'Actual Query ID', 'Expected Query ID'):
        display_df[col] = display_df[col].where(display_df[col].notna(), '-').astype(str)
    st.dataframe(display_df)

    st.markdown("#### Test case details")
    for i, result in enumerate(results):
        question = get_test_question(ground_truth, i)
        with st.expander(f"TC{i+1:03d}: {question}"):
            if 'Expected Answer' in ground_truth.columns:
                st.caption(f"Ground truth: {md_safe(ground_truth['Expected Answer'].iloc[i])}")
            if result.get('log') and result['log'].get('user_question'):
                render_result(result, show_trace=True, key_prefix=f"eval_{i}")
            else:
                st.error(result.get('narrative'))


# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    st.set_page_config(page_title="Credit Risk Query Engine", page_icon="🏦", layout="wide")
    init_state()

    st.title("🏦 Northbridge Bank - Credit Risk Query Engine")
    st.caption("Natural-language analytics over the commercial lending portfolio, with verified queries and a validation gate.")

    api_key, _ = get_openai_credentials()
    db_file = resolve_path(DB_PATH)
    gt_file = find_test_queries_file()

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Configuration")
        st.write(f"{'OK' if api_key else 'MISSING'}: OpenAI API key")
        st.write(f"{'OK' if db_file.exists() else 'MISSING'}: database `{db_file.name}`")
        st.write(f"{'OK' if gt_file else 'MISSING (optional)'}: test queries CSV")
        st.write(f"Model: `{MODEL_NAME}` (temperature 0)")
        show_trace = st.checkbox("Show pipeline trace", value=True)

        st.header("How it works")
        st.markdown(
            "1. **Classify** the question: verified template or generated SQL\n"
            "2. **Build** the SQL from the library or the LLM\n"
            "3. **Validate**: read-only, schema, EXPLAIN, LLM relevance, template integrity\n"
            "4. **Retry** once if generated SQL fails validation\n"
            "5. **Execute** on a read-only connection\n"
            "6. **Summarise** the result in business language\n"
            "7. **Escalate** to a human analyst if it still fails"
        )

        if st.session_state["history"]:
            st.header("Session history")
            st.dataframe(pd.DataFrame(st.session_state["history"]))

    # ---- Preconditions ----
    problems = []
    if not api_key:
        problems.append(
            "`OPENAI_API_KEY` is not set. Add it to `.streamlit/secrets.toml`, export it as an "
            "environment variable, or provide a `config.json`."
        )
    if not db_file.exists():
        problems.append(
            f"Database file not found at `{db_file}`. Place `credit_risk_portfolio.db` next to "
            "`app.py` or set `DB_PATH`."
        )
    if problems:
        for p in problems:
            st.error(p)
        st.stop()

    conn = get_connection(str(db_file))
    ground_truth = load_ground_truth(str(gt_file)) if gt_file else None

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Ask a question", "Verified query library", "Schema and data", "Evaluation"]
    )
    with tab1:
        tab_ask(conn, ground_truth, show_trace)
    with tab2:
        tab_library(conn)
    with tab3:
        tab_schema(conn)
    with tab4:
        tab_evaluation(conn, ground_truth)


if __name__ == "__main__":
    main()

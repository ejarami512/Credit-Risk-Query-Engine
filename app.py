"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb

Pipeline (unchanged from the notebook):
    classify_intent -> (verified template | generate_query) -> validate_query
    -> (retry_generation once on generated track) -> execute_query -> generate_response
    -> escalate to a human analyst if validation still fails.

Run locally:   streamlit run app.py
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Configuration (secrets / environment variables / local files)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


def _get_setting(name, default=None):
    """Read a setting from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


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


def get_openai_credentials():
    cfg = _load_config_json()
    api_key = _get_setting("OPENAI_API_KEY") or cfg.get("OPENAI_API_KEY")
    base_url = (
        _get_setting("OPENAI_BASE_URL")
        or _get_setting("OPENAI_API_BASE")
        or cfg.get("OPENAI_API_BASE")
        or cfg.get("OPENAI_BASE_URL")
    )
    return api_key, base_url


MODEL_NAME = _get_setting("OPENAI_MODEL", "gpt-4o-mini")
DB_PATH = _get_setting("DB_PATH", "credit_risk_portfolio.db")
TEST_QUERIES_PATH = _get_setting("TEST_QUERIES_PATH", "test_queries.csv")
AUDIT_LOG_PATH = _get_setting("AUDIT_LOG_PATH", "audit_log.jsonl")


def _resolve_path(p):
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


# ----------------------------------------------------------------------------
# LLM initialisation (same models/temperature as the notebook)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _build_llm(model_name, role):
    """role: 'generator' or 'evaluator' (two separate instances, as in the notebook)."""
    api_key, base_url = get_openai_credentials()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to .streamlit/secrets.toml, "
            "set it as an environment variable, or provide a config.json."
        )
    os.environ["OPENAI_API_KEY"] = str(api_key)
    if base_url:
        os.environ["OPENAI_BASE_URL"] = str(base_url)
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
        base_url=base_url or None,
    )


def get_llm():
    return _build_llm(MODEL_NAME, "generator")


def get_evaluator_llm():
    return _build_llm(MODEL_NAME, "evaluator")


# ----------------------------------------------------------------------------
# Database loading (read-only connection)
# ----------------------------------------------------------------------------
def open_connection():
    """Open a READ-ONLY SQLite connection (URI mode with ?mode=ro)."""
    db_file = _resolve_path(DB_PATH)
    if not db_file.exists():
        raise FileNotFoundError(
            f"Database file not found: {db_file}. Place credit_risk_portfolio.db next to "
            "app.py or set DB_PATH."
        )
    return sqlite3.connect(f"{db_file.as_uri()}?mode=ro", uri=True, check_same_thread=False)


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


# ----------------------------------------------------------------------------
# Ground-truth test cases (test_queries.csv)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_ground_truth(path_str):
    path = _resolve_path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


# ----------------------------------------------------------------------------
# Verified Query Template Library (SQL kept exactly as in the notebook)
# ----------------------------------------------------------------------------
# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """
SELECT
    s.sector_name,
    SUM(l.total_outstanding) / 1000000.0 AS total_outstanding,
    SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN l.total_outstanding ELSE 0 END) / 1000000.0 AS npa_exposure
FROM loan_master l
INNER JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding DESC;
"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """
SELECT
    loan_category,
    COUNT(loan_account_number) AS loan_count,
    SUM(total_outstanding) / 1000000.0 AS total_outstanding
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding DESC;
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


# ----------------------------------------------------------------------------
# Tool definitions
# ----------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Returns:
    - dict: 'route' (verified or generated), 'query_id' (template ID or None),
            'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
System: You are an expert intent classifier for a credit risk portfolio database. Your job is to classify a user's natural language question and route it to either "verified" or "generated".

If the user's question maps directly to one of the pre-approved Verified Queries listed below, you MUST select "verified" and return the corresponding query ID.
A query matches if it asks for the same analytical metrics, group, and structure, even if there are slight wording differences or minor filter details (e.g. "by sector", "across all sectors", "outstanding and NPA summary").

Pre-approved Verified Queries:
{library_descriptions}

If the user's question does NOT match any of these pre-approved query descriptions, you must select "generated" and return null for query_id.

User Question:
{user_question}

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
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.
    '''

    generation_prompt = f"""
System: You are an expert SQL engineer. Generate a single, valid, read-only SQLite SELECT query to answer the user's question.
Do not write any introductory or explanatory text. Return ONLY the raw SQL statement.

Guidelines:
1. Use correct SQLite syntax.
2. Ensure all referenced tables and columns exist in the database schema below.
3. Make sure to divide large exposures or totals by 1,000,000.0 to convert them to millions if requested, or present raw numbers otherwise.
4. Ensure the query is strictly read-only.

Database Schema:
{schema_context}

User Question:
{user_question}
"""

    sql = get_llm().invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


def _strip_trailing_semicolons(sql):
    return sql.strip().rstrip(';').strip()


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
    if ';' in candidate_sql.rstrip(';').rstrip():
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
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]
    # Same behaviour as the notebook: unknown identifiers are not a hard failure
    # (aliases and string literals would cause false positives). They are surfaced
    # in the audit details for analyst review.
    result['unknown_identifiers'] = sorted(set(unknown))

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
            # Wrapped in a sub-select so the trailing ';' of the stored templates does not
            # break the "LIMIT 0" probe (sqlite3 allows only one statement per execute()).
            expected_cols = [d[0] for d in cur.execute(
                f"SELECT * FROM ({_strip_trailing_semicolons(expected_sql)}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(
                f"SELECT * FROM ({_strip_trailing_semicolons(candidate_sql)}) LIMIT 0").description]
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
"""

    revised_sql = get_llm().invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame
    together with a reasonableness check. Purely deterministic (no LLM).
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
"""

    narrative = get_llm().invoke(response_prompt).content.strip()
    return narrative


# ----------------------------------------------------------------------------
# Pipeline orchestration
# ----------------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

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
        'narrative': None,
        'warnings': [],
        'trace': []
    }

    def note(message):
        log['trace'].append(message)
        if verbose:
            print(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    note(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    note(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    note(f"[2] Query Construction: {'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    note(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        note(f"    Failed check: {gate.get('failed_check')}")
        note(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        note(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        note(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            note(f"    Retry failed check: {gate.get('failed_check')}")
            note(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        note(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = exec_result['warnings']

    note(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        note(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    note(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ----------------------------------------------------------------------------
# Audit trail
# ----------------------------------------------------------------------------
def record_audit(log, source="interactive"):
    """Keep an audit record in the session and append it to a JSON-lines file."""
    gate = log.get('gate_result') or {}
    entry = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source': source,
        'user_question': log.get('user_question'),
        'route': log.get('route'),
        'query_id': log.get('query_id'),
        'match_reason': log.get('match_reason'),
        'retry_used': log.get('retry_used'),
        'escalated': log.get('escalated'),
        'validation_passed': gate.get('passed'),
        'failed_check': gate.get('failed_check'),
        'validation_details': gate.get('details'),
        'unknown_identifiers': gate.get('unknown_identifiers'),
        'confidence': log.get('confidence'),
        'candidate_sql': log.get('candidate_sql'),
        'executed_sql': log.get('executed_sql'),
        'row_count': log.get('row_count'),
        'warnings': log.get('warnings'),
        'narrative': log.get('narrative'),
    }
    st.session_state.setdefault('audit_log', []).append(entry)
    try:
        with open(_resolve_path(AUDIT_LOG_PATH), 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, default=str) + '\n')
    except Exception:
        # Filesystem may be read-only / ephemeral (e.g. hosted deployments).
        st.session_state['audit_file_error'] = True
    return entry


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def fmt_confidence(value):
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value) if value is not None else "N/A"


def safe_markdown(text):
    """Escape '$' so dollar amounts are not rendered as LaTeX."""
    return (text or "").replace("$", r"\$")


def render_result(result, key_prefix="main"):
    log = result['log']
    df = result.get('dataframe')

    if log['escalated']:
        st.error("⚠️ Escalated to a human analyst")
        st.markdown(safe_markdown(log['narrative']))
    else:
        st.subheader("Answer")
        st.markdown(safe_markdown(log['narrative']))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Route", str(log['route']).upper())
    c2.metric("Verified Query ID", log['query_id'] if log['query_id'] else "—")
    c3.metric("Confidence", fmt_confidence(log['confidence']))
    c4.metric("Rows returned", log['row_count'] if log['row_count'] is not None else "—")
    c5.metric("Retry used", "Yes" if log['retry_used'] else "No")

    if log.get('match_reason'):
        st.caption(f"Routing reason: {log['match_reason']}")

    for w in log.get('warnings') or []:
        st.warning(w)

    sql_label = "SQL query used" if not log['escalated'] else "Candidate SQL (failed validation)"
    with st.expander(sql_label, expanded=True):
        st.code((log['executed_sql'] or log['candidate_sql'] or '').strip(), language='sql')

    if df is not None:
        with st.expander("Raw data returned", expanded=True):
            st.dataframe(df)
            st.download_button(
                "Download result as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="query_result.csv",
                mime="text/csv",
                key=f"{key_prefix}_download_csv",
            )

    with st.expander("Validation details"):
        st.json(log['gate_result'] or {})

    with st.expander("Pipeline trace"):
        st.code("\n".join(log['trace']), language='text')


def normalise_expected_id(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return None if s.lower() in ('', 'none', 'null', 'nan', 'n/a', 'na', '-') else s.upper()


FALLBACK_EXPECTED_ROUTES = ['verified', 'verified', 'generated', 'generated', 'generated']
FALLBACK_EXPECTED_IDS = ['VQ1', 'VQ8', None, None, None]
DEFAULT_SAMPLE_QUESTIONS = [
    "Show me the sector-wise outstanding and NPA breakdown",
    "What is the DPD aging profile of the portfolio?",
    "What is the average interest rate by sector?",
]


def run_evaluation(ground_truth, n_cases, db_connection):
    """Evaluate the engine against test_queries.csv (same metrics as the notebook)."""
    q_col = 'User Query' if 'User Query' in ground_truth.columns else 'question'
    rows, details = [], []
    progress = st.progress(0.0, text="Running test cases...")

    for i in range(n_cases):
        question = ground_truth[q_col].iloc[i]

        if 'Expected Route' in ground_truth.columns:
            expected_route = str(ground_truth['Expected Route'].iloc[i]).strip().lower()
        else:
            expected_route = FALLBACK_EXPECTED_ROUTES[i] if i < len(FALLBACK_EXPECTED_ROUTES) else None

        if 'Expected Query ID' in ground_truth.columns:
            expected_id = normalise_expected_id(ground_truth['Expected Query ID'].iloc[i])
        else:
            expected_id = FALLBACK_EXPECTED_IDS[i] if i < len(FALLBACK_EXPECTED_IDS) else None

        test_id = ground_truth['Test Case'].iloc[i] if 'Test Case' in ground_truth.columns else f'TC{i + 1:03d}'

        try:
            tr = run_pipeline(question, db_connection, verified_query_library, database_schema, verbose=False)
            record_audit(tr['log'], source="evaluation")
        except Exception as exc:
            rows.append({
                'Test Case': test_id, 'User Query': question,
                'Expected Route': expected_route, 'Actual Route': 'ERROR', 'Route Match': False,
                'Expected Query ID': expected_id, 'Actual Query ID': None, 'Query ID Match': False,
                'Confidence': None, 'Rows Returned': None,
            })
            details.append({'test_id': test_id, 'error': str(exc)})
            progress.progress((i + 1) / n_cases, text=f"Completed {i + 1}/{n_cases}")
            continue

        actual_route = tr.get('route')
        actual_id = tr.get('query_id')
        rows.append({
            'Test Case': test_id,
            'User Query': question,
            'Expected Route': expected_route,
            'Actual Route': actual_route,
            'Route Match': actual_route == expected_route,
            'Expected Query ID': expected_id,
            'Actual Query ID': actual_id,
            'Query ID Match': (expected_id is None and actual_id is None) or (actual_id == expected_id),
            'Confidence': tr.get('confidence'),
            'Rows Returned': tr.get('row_count'),
        })
        details.append({
            'test_id': test_id,
            'result': tr,
            'expected_answer': ground_truth['Expected Answer'].iloc[i] if 'Expected Answer' in ground_truth.columns else None,
        })
        progress.progress((i + 1) / n_cases, text=f"Completed {i + 1}/{n_cases}")

    progress.empty()
    evaluation_df = pd.DataFrame(rows)

    path_accuracy = evaluation_df['Route Match'].mean() * 100
    verified_mask = evaluation_df['Expected Route'] == 'verified'
    query_accuracy = (evaluation_df.loc[verified_mask, 'Query ID Match'].mean() * 100
                      if verified_mask.any() else float('nan'))
    numeric_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce')
    average_confidence = numeric_confidence.mean()

    return {
        'df': evaluation_df,
        'details': details,
        'path_accuracy': path_accuracy,
        'query_accuracy': query_accuracy,
        'average_confidence': average_confidence,
    }


# ----------------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------------
def main():
    st.title("🏦 Northbridge Bank - Credit Risk Query Engine")
    st.caption(
        "Ask routine commercial lending portfolio questions in plain English. Recurring questions use "
        "pre-approved, version-controlled SQL templates; other questions get generated read-only SQL that "
        "must pass validation. Every answer shows the SQL used, the raw data, and a confidence score, "
        "and is logged to an audit trail."
    )

    api_key, _ = get_openai_credentials()
    ground_truth = load_ground_truth(TEST_QUERIES_PATH)

    # ---------------------------- Sidebar ----------------------------
    with st.sidebar:
        st.header("System status")
        st.write(f"**LLM model:** `{MODEL_NAME}`")
        db_file = _resolve_path(DB_PATH)
        if db_file.exists():
            st.success("Database found (opened read-only)")
        else:
            st.error(f"Database not found: {db_file.name}")
        if api_key:
            st.success("OpenAI credentials configured")
        else:
            st.error("OPENAI_API_KEY not configured")
        if ground_truth is None:
            st.info("test_queries.csv not found - Evaluation tab disabled")
        st.divider()
        st.markdown(
            "**Safeguards**\n"
            "- Read-only database connection\n"
            "- SELECT / WITH statements only\n"
            "- One automatic retry, then human escalation\n"
            "- Full audit trail"
        )

    tab_ask, tab_library, tab_eval, tab_audit, tab_schema = st.tabs(
        ["💬 Ask a Question", "📚 Verified Query Library", "🧪 Evaluation", "🧾 Audit Trail", "🗄️ Database & Schema"]
    )

    # ---------------------------- Ask tab ----------------------------
    with tab_ask:
        placeholder_choice = "— choose a sample question —"
        if ground_truth is not None and ('User Query' in ground_truth.columns or 'question' in ground_truth.columns):
            q_col = 'User Query' if 'User Query' in ground_truth.columns else 'question'
            sample_questions = ground_truth[q_col].dropna().astype(str).tolist()
        else:
            sample_questions = DEFAULT_SAMPLE_QUESTIONS

        def _use_sample():
            choice = st.session_state.get('sample_choice')
            if choice and choice != placeholder_choice:
                st.session_state['question_input'] = choice

        st.selectbox(
            "Sample questions",
            [placeholder_choice] + sample_questions,
            key="sample_choice",
            on_change=_use_sample,
        )
        user_question = st.text_area(
            "Your question",
            key="question_input",
            height=100,
            placeholder="e.g. Show me the sector-wise outstanding and NPA breakdown",
        )

        run_clicked = st.button("Run query", type="primary", disabled=not api_key)
        if not api_key:
            st.warning("Configure OPENAI_API_KEY to enable the query engine (see the README / secrets setup).")

        if run_clicked:
            if not user_question or not user_question.strip():
                st.warning("Please enter a question.")
            else:
                conn = None
                try:
                    conn = open_connection()
                    with st.spinner("Running the query engine pipeline..."):
                        result = run_pipeline(
                            user_question.strip(), conn, verified_query_library,
                            database_schema, verbose=False,
                        )
                    record_audit(result['log'])
                    st.session_state['last_result'] = result
                except Exception as exc:
                    st.session_state.pop('last_result', None)
                    st.error(f"The pipeline failed: {exc}")
                finally:
                    if conn is not None:
                        conn.close()

        if st.session_state.get('last_result'):
            st.divider()
            render_result(st.session_state['last_result'], key_prefix="ask")

    # ---------------------------- Library tab ----------------------------
    with tab_library:
        st.subheader("Verified Query Template Library")
        st.write(f"{len(verified_query_library)} pre-approved SQL templates for recurring questions.")
        for qid, entry in verified_query_library.items():
            with st.expander(f"{qid}: {entry['description']}"):
                st.code(entry['sql'].strip(), language='sql')

    # ---------------------------- Evaluation tab ----------------------------
    with tab_eval:
        st.subheader("Evaluation against ground truth")
        if ground_truth is None:
            st.info("Add test_queries.csv next to app.py (or set TEST_QUERIES_PATH) to enable evaluation.")
        else:
            st.write("Compares the route and verified query selected by the engine with the expected values.")
            st.dataframe(ground_truth)
            max_cases = len(ground_truth)
            n_cases = st.number_input(
                "Number of test cases to run (from the top of the file)",
                min_value=1, max_value=max_cases, value=min(5, max_cases), step=1,
            )
            if st.button("Run evaluation", disabled=not api_key):
                conn = None
                try:
                    conn = open_connection()
                    st.session_state['evaluation'] = run_evaluation(ground_truth, int(n_cases), conn)
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")
                finally:
                    if conn is not None:
                        conn.close()

            ev = st.session_state.get('evaluation')
            if ev:
                m1, m2, m3 = st.columns(3)
                m1.metric("Selected Path Accuracy", f"{ev['path_accuracy']:.1f}%")
                m2.metric("Selected Query Accuracy",
                          "N/A" if pd.isna(ev['query_accuracy']) else f"{ev['query_accuracy']:.1f}%")
                m3.metric("Average Confidence Score",
                          "N/A" if pd.isna(ev['average_confidence']) else f"{ev['average_confidence']:.2f}")
                st.dataframe(ev['df'])

                st.markdown("**Per-test-case output**")
                for d in ev['details']:
                    with st.expander(str(d['test_id'])):
                        if 'error' in d:
                            st.error(d['error'])
                            continue
                        if d.get('expected_answer') is not None and not pd.isna(d['expected_answer']):
                            st.markdown(f"**Expected answer:** {safe_markdown(str(d['expected_answer']))}")
                        render_result(d['result'], key_prefix=f"eval_{d['test_id']}")

    # ---------------------------- Audit tab ----------------------------
    with tab_audit:
        st.subheader("Audit trail (this session)")
        audit = st.session_state.get('audit_log', [])
        if not audit:
            st.info("No queries have been run in this session yet.")
        else:
            audit_df = pd.DataFrame(audit)
            st.dataframe(audit_df)
            c1, c2 = st.columns(2)
            c1.download_button("Download audit log (CSV)", audit_df.to_csv(index=False).encode('utf-8'),
                               file_name="audit_log.csv", mime="text/csv")
            c2.download_button("Download audit log (JSON)", json.dumps(audit, indent=2, default=str).encode('utf-8'),
                               file_name="audit_log.json", mime="application/json")
        if st.session_state.get('audit_file_error'):
            st.warning("Could not write the persistent audit file on this host; "
                       "the session audit log above is still available for download.")
        else:
            st.caption(f"Each entry is also appended to `{AUDIT_LOG_PATH}` (JSON lines).")

    # ---------------------------- Schema tab ----------------------------
    with tab_schema:
        st.subheader("Database schema provided to the LLM")
        st.code(database_schema.strip(), language='text')
        st.subheader("Table previews")
        conn = None
        try:
            conn = open_connection()
            for table_name in ['sector_master', 'loan_master', 'borrower_rating', 'provisioning']:
                with st.expander(table_name):
                    st.dataframe(pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn))
        except Exception as exc:
            st.error(f"Could not load table previews: {exc}")
        finally:
            if conn is not None:
                conn.close()


main()

import pandas as pd
import re
from datetime import datetime, timedelta
from groq import Groq
import numpy as np
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging
import io
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq client with API key from environment variable
try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    client = Groq(api_key=api_key)
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    client = None

# Configuration
CONFIG = {
    "email_validation": {
        "model": "llama3-70b-8192",
        "max_retries": 3,
        "retry_delay": 1,
        "batch_size": 10,
        "rate_limit_delay": 0.5
    },
    "dummy_patterns": {
        'noname', 'none', 'noemail', 'example', 'optout',
        'evenflow', 'noreply', 'no-reply', 'test', 'invalid',
        'fake', 'dummy', 'placeholder', 'temp', 'spam',
        'no@email', 'nomail', 'void', 'blank', 'unknown'
    },
    "suspicious_domains": {
        'mailinator.com', 'example.com', 'test.com',
        'trashmail.com', 'guerrillamail.com',
        'email.com', 'noemail.com', 'nomail.com',
        'void.com', 'blank.com', 'unknown.com'
    }
}

def get_yesterdays_date():
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y%m%d")

@lru_cache(maxsize=10000)
def email_hash(email):
    return hashlib.md5(email.lower().encode()).hexdigest()

def validate_email_with_ai(email):
    if not client:
        logger.warning("No Groq client available, marking email as invalid")
        return False

    for attempt in range(CONFIG["email_validation"]["max_retries"]):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an advanced email validation system. Analyze this email address and determine if it's "
                            "a legitimate email or a dummy/placeholder address. Consider:\n"
                            "1. Common dummy email patterns (like NO@EMAIL.COM, no@email.com, no-reply@domain.com)\n"
                            "2. Suspicious domains (like example.com, test.com, mailinator.com)\n"
                            "3. Unusual formatting or patterns\n"
                            "4. Any email that looks intentionally fake or temporary\n"
                            "\n"
                            "Key indicators of dummy emails:\n"
                            "- Contains words like 'no', 'none', 'example', 'test', 'fake', 'dummy'\n"
                            "- Uses domains known for temporary emails\n"
                            "- Appears to be a placeholder rather than a real person's email\n"
                            "\n"
                            "Respond ONLY with 'True' if the email appears legitimate or 'False' if it appears to be a dummy."
                        )
                    },
                    {"role": "user", "content": f"Email to analyze: {email}"}
                ],
                model=CONFIG["email_validation"]["model"],
                temperature=0.1,
                max_tokens=1
            )
            result = response.choices[0].message.content.strip().lower()
            logger.debug(f"Email {email} validated as {result}")
            return result == 'true'
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {email}: {e}")
            if attempt < CONFIG["email_validation"]["max_retries"] - 1:
                time.sleep(CONFIG["email_validation"]["retry_delay"])
                continue
            logger.error(f"AI validation failed for {email} after retries: {e}")
            return False

def batch_validate_emails(emails, progress_callback=None):
    valid_emails = {}

    def process_batch(batch):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(validate_single_email, batch))
        return dict(zip(batch, results))

    def validate_single_email(email):
        if not email or pd.isna(email):
            logger.debug(f"Email {email} is null or NaN, marking as invalid")
            return False
        email_lower = email.lower()
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email_lower):
            logger.debug(f"Email {email} failed regex validation")
            return False
        if any(pattern in email_lower for pattern in CONFIG["dummy_patterns"]):
            logger.debug(f"Email {email} contains dummy pattern")
            return False
        domain = email_lower.split('@')[-1]
        if domain in CONFIG["suspicious_domains"]:
            logger.debug(f"Email {email} has suspicious domain")
            return False
        # Fallback to True if AI validation fails but regex passes
        result = validate_email_with_ai(email)
        if result is False and not any(pattern in email_lower for pattern in CONFIG["dummy_patterns"]) and domain not in CONFIG["suspicious_domains"]:
            logger.debug(f"Email {email} passed regex but failed AI, allowing as valid")
            return True
        return result

    total_emails = len(emails)
    for i in range(0, total_emails, CONFIG["email_validation"]["batch_size"]):
        batch = emails[i:i + CONFIG["email_validation"]["batch_size"]]
        valid_emails.update(process_batch(batch))
        if progress_callback:
            progress_callback(i + len(batch), total_emails)
        time.sleep(CONFIG["email_validation"]["rate_limit_delay"])
    
    return valid_emails

def clean_phone(phone):
    if pd.isna(phone) or not isinstance(phone, (str, int, float)):
        logger.debug(f"Phone {phone} is null or invalid type, returning NaN")
        return np.nan
    cleaned = re.sub(r'[^0-9]', '', str(phone))
    if len(cleaned) == 10:
        logger.debug(f"Phone {phone} cleaned to {cleaned}")
        return f"({cleaned[:3]}) {cleaned[3:6]}-{cleaned[6:]}"
    elif len(cleaned) == 11 and cleaned[0] == '1':
        logger.debug(f"Phone {phone} cleaned to {cleaned[1:]}")
        return f"({cleaned[1:4]}) {cleaned[4:7]}-{cleaned[7:]}"
    logger.debug(f"Phone {phone} invalid after cleaning: {cleaned}")
    return np.nan

def load_data_safely(df, required_columns):
    """
    Load DataFrame and dynamically detect header row.
    """
    try:
        # Read the Excel file as a raw DataFrame to inspect rows
        raw_df = df

        # Find the header row by checking for a row with at least half of the required columns
        for row_idx in range(min(20, len(raw_df))):  # Check up to first 20 rows
            potential_headers = raw_df.iloc[row_idx].astype(str).str.strip().str.lower()
            matched_columns = set(potential_headers) & set(col.lower() for col in required_columns)
            if len(matched_columns) >= len(required_columns) * 0.5:  # At least 50% match
                # Set headers and load data from this row
                raw_df.columns = potential_headers
                df = raw_df.iloc[row_idx + 1:].reset_index(drop=True)
                df.columns = df.columns.str.strip().str.lower()
                break
        else:
            raise ValueError("Could not identify header row with required columns")

        # Verify required columns exist
        missing = set(col.lower() for col in required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        return df
    except Exception as e:
        logger.error(f"Error processing DataFrame: {e}")
        return pd.DataFrame(columns=required_columns)

def remove_duplicates(df, key_columns):
    return df.drop_duplicates(subset=key_columns, keep='first')

def process_data_pipeline(no_shows_df, planned_next_df, prior_appointments_df, prior_repairs_df):
    logger.info("Starting no-show processing pipeline...")
    
    data_files = {
        'no_shows': (no_shows_df, ['service center', 'planned date', 'customer', 'dms_id', 'vin', 'customer email', 'customer phone', 'reporting_status', 'customer_id']),
        'planned_next': (planned_next_df, ['sc_name', 'planned date', 'dms_id', 'customer', 'vin', 'customer phone', 'customer email', 'reporting_status']),
        'prior_appointments': (prior_appointments_df, ['dealer', 'planned date', 'dms_id', 'customer', 'vin', 'customer phone', 'customer email', 'status']),
        'prior_repairs': (prior_repairs_df, ['sc_name', 'open_date', 'vin', 'customer', 'customer phone', 'customer email'])
    }
    
    # We need to dynamically find the correct columns from the files
    # For planned_next, we look for 'reporting_status'.
    planned_next_cols = ['sc_name', 'planned date', 'dms_id', 'customer', 'vin', 'customer phone', 'customer email']
    if 'reporting_status' in planned_next_df.columns.str.lower():
        planned_next_cols.append('reporting_status')
    data_files['planned_next'] = (planned_next_df, planned_next_cols)
    
    # For prior_appointments, we look for 'status'.
    prior_appts_cols = ['dealer', 'planned date', 'dms_id', 'customer', 'vin', 'customer phone', 'customer email']
    if 'status' in prior_appointments_df.columns.str.lower():
        prior_appts_cols.append('status')
    data_files['prior_appointments'] = (prior_appointments_df, prior_appts_cols)

    dfs = {name: load_data_safely(df, cols) for name, (df, cols) in data_files.items()}
    
    df_no_shows = dfs['no_shows'].copy()
    df_no_shows = df_no_shows[df_no_shows['vin'].notna() & (df_no_shows['vin'] != '')]
    logger.info(f"After VIN filtering, df_no_shows has {len(df_no_shows)} rows")
    logger.debug(f"No-show VINs: {df_no_shows['vin'].tolist()}")
    
    # --- START OF CHANGE ---
    # The original logic only excluded VINs if their status was 'showed'.
    # This is updated to exclude VINs for any status indicating they have an active
    # or rescheduled appointment, or have already shown up.
    
    # These are statuses that mean the customer has re-engaged and should be excluded.
    positive_engagement_statuses = ['active', 'rescheduled', 'showed', 'walk_in_with_appointment']
    
    exclusion_sources = {
        'planned_next': ('vin', 'reporting_status', positive_engagement_statuses),
        'prior_appointments': ('vin', 'status', positive_engagement_statuses),
        'prior_repairs': ('vin', None, None)  # No status column, so any entry is an exclusion
    }

    for source, (col, status_col, status_values_to_exclude) in exclusion_sources.items():
        if not dfs[source].empty:
            # Check if we need to filter by status
            if status_col and status_values_to_exclude and status_col in dfs[source].columns:
                # Exclude VINs where the status is one of the positive engagement types.
                # .astype(str).str.lower() handles non-string data and ensures case-insensitivity.
                exclude_vins = dfs[source][
                    dfs[source][status_col].astype(str).str.lower().isin(status_values_to_exclude)
                ][col].dropna().unique()
            else:
                # For prior_repairs or if status column doesn't exist, exclude all VINs.
                exclude_vins = dfs[source][col].dropna().unique()
            
            logger.info(f"Excluding {len(exclude_vins)} VINs from {source}")
            logger.debug(f"{source} excluded VINs: {exclude_vins.tolist()}")
            df_no_shows = df_no_shows[~df_no_shows['vin'].isin(exclude_vins)]
    # --- END OF CHANGE ---
    
    logger.info(f"After exclusion, df_no_shows has {len(df_no_shows)} rows")
    
    df_clean = df_no_shows.sort_values(by=['service center', 'customer'])
    logger.info(f"df_clean has {len(df_clean)} rows")
    
    logger.info("Validating emails with AI-powered detection...")
    unique_emails = df_clean['customer email'].dropna().unique()
    logger.info(f"Found {len(unique_emails)} unique emails")
    email_validation_map = batch_validate_emails(unique_emails)
    df_clean['email_valid'] = df_clean['customer email'].map(email_validation_map)
    logger.info(f"Valid emails: {df_clean['email_valid'].sum()} out of {len(df_clean)}")
    
    logger.info("Cleaning phone numbers...")
    df_clean['phone_clean'] = df_clean['customer phone'].apply(clean_phone)
    logger.info(f"Valid phone numbers: {df_clean['phone_clean'].notna().sum()} out of {len(df_clean)}")
    
    logger.info("Generating final output lists...")
    
    df_email = df_clean[(df_clean['email_valid'] == True) & (df_clean['customer email'].notna()) & (df_clean['customer email'] != '')].copy()
    df_email['first_name'] = df_email['customer'].str.split().str[0]
    df_email = df_email[['first_name', 'customer email', 'service center', 'vin']]
    df_email.rename(columns={'customer email': 'email', 'service center': 'sc_name'}, inplace=True)
    df_email = remove_duplicates(df_email, ['vin']).drop(columns=['vin'])
    logger.info(f"df_email has {len(df_email)} rows")
    
    df_text = df_clean[(df_clean['phone_clean'].notna()) & (df_clean['customer phone'].notna()) & (df_clean['customer phone'] != '')].copy()
    df_text['first_name'] = df_text['customer'].str.split().str[0]
    df_text = df_text[['first_name', 'phone_clean', 'service center', 'vin']]
    df_text.rename(columns={'phone_clean': 'phone', 'service center': 'sc_name'}, inplace=True)
    df_text['address_country'] = 'US'
    df_text = remove_duplicates(df_text, ['vin']).drop(columns=['vin'])
    logger.info(f"df_text has {len(df_text)} rows")
    
    df_target = df_clean[(df_clean['email_valid'] == True) & (df_clean['customer email'].notna()) & (df_clean['customer email'] != '') &
                         (df_clean['phone_clean'].notna()) & (df_clean['customer phone'].notna()) & (df_clean['customer phone'] != '')].copy()
    df_target = df_target[['service center', 'planned date', 'customer', 'dms_id', 'vin', 'customer email', 'customer phone', 'reporting_status', 'customer_id']]
    df_target.rename(columns={'customer email': 'email', 'customer phone': 'phone'}, inplace=True)
    df_target = remove_duplicates(df_target, ['vin'])
    logger.info(f"df_target has {len(df_target)} rows")
    
    yesterdays_date = get_yesterdays_date()
    current_time = datetime.now().strftime("%H%M%S")
    output_filename = f"No_Show_Target_Lists_{yesterdays_date}_{current_time}.xlsx"
    
    # Use BytesIO to generate the file in memory
    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        df_email.to_excel(writer, sheet_name='No Show Email Target List', index=False)
        df_text.to_excel(writer, sheet_name='No Show Text Target List', index=False)
        df_target.to_excel(writer, sheet_name='No Show Target Lists', index=False)
    
    output_buffer.seek(0)
    logger.info("Processing complete. Output generated in memory.")
    return output_buffer, output_filename

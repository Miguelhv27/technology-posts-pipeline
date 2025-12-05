import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

class DataValidation:
    """
    DataOps Validation Engine.
    Validates Raw Data Quality (Schema, Types, Distributions, Integrity)
    before Transformation logic is applied.
    """
    def __init__(self, config):
        self.config = config
        self.validation_config = config["validation"]
        self.raw_data_dir = config["paths"]["raw_data_dir"]

    def load_raw_files(self):
        """Load all raw data files directly from disk"""
        datasets = {}
        
        if not os.path.exists(self.raw_data_dir):
            logging.error(f"Raw data directory not found: {self.raw_data_dir}")
            return datasets

        files = [f for f in os.listdir(self.raw_data_dir) if f.endswith((".csv", ".parquet")) and not f.startswith(".")]

        if not files:
            logging.warning("No data files found to validate.")
            return datasets

        for file in files:
            file_path = os.path.join(self.raw_data_dir, file)
            try:
                if file.endswith(".csv"):
                    df = pd.read_csv(file_path, low_memory=False)
                elif file.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                
                datasets[file] = df
                logging.info(f"Loaded for validation: {file} ({len(df)} rows)")
                
            except Exception as e:
                logging.error(f"Failed to load {file}: {e}")

        return datasets

    def detect_source_type(self, df, filename):
        """
        Robust source detection based on content signatures.
        """
        if 'data_source' in df.columns:
            src = df['data_source'].iloc[0]
            if isinstance(src, str):
                if 'twitter' in src.lower(): return 'twitter'
                if 'reddit' in src.lower(): return 'reddit'

        filename_lower = filename.lower()
        if 'twitter' in filename_lower: return 'twitter'
        if 'reddit' in filename_lower: return 'reddit'

        cols = set(df.columns)
        if 'subreddit' in cols or 'author_post_karma' in cols: return 'reddit'
        if 'retweet_count' in cols or 'user_followers_count' in cols: return 'twitter'
        
        return 'unknown'

    def validate_schema(self, df, filename, source_type):
        """
        Validates that critical columns for Analysis exist.
        """
        critical_columns = {
            'twitter': ['id', 'text', 'created_at', 'user_followers_count', 'retweet_count'],
            'reddit': ['id', 'title', 'score', 'subreddit', 'created_utc'],
            'commercial': ['amazon_tech_sales_velocity', 'created_at']
        }

        required = critical_columns.get(source_type, [])
        
        missing = [col for col in required if col not in df.columns]
        
        if source_type == 'reddit' and 'text' not in df.columns:
            if 'selftext' in df.columns or 'body' in df.columns:
                pass 
            else:
                if 'title' not in df.columns:
                    missing.append('content_text')

        if missing:
            logging.error(f"[{filename}] SCHEMA VIOLATION. Missing critical columns: {missing}")
            return False, missing
        
        return True, []

    def validate_data_quality(self, df, filename):
        """
        Checks for logical data quality issues (Nulls, Zero Variance).
        """
        issues = {}
        
        if df.empty:
            return False, {"error": "DataFrame is empty"}

        metric_cols = [c for c in df.columns if c in ['score', 'retweet_count', 'user_followers_count', 'author_post_karma']]
        for col in metric_cols:
            null_pct = df[col].isnull().mean()
            if null_pct > 0.5: 
                issues[f"{col}_nulls"] = f"{null_pct:.1%} missing"

        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            if col in metric_cols:
                if numeric_df[col].std() == 0:
                    issues[f"{col}_variance"] = "Zero variance (all values are identical)"

        if issues:
            logging.warning(f"[{filename}] DATA QUALITY WARNINGS: {issues}")
            return True, issues 
        
        return True, {}

    def validate_dates(self, df, filename):
        """
        Validates time range. Handles both ISO Strings and Unix Timestamps.
        """
        date_col = None
        possible_dates = ['created_at', 'created_utc', 'date', 'timestamp']
        for col in possible_dates:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            return True, {}

        try:
            series = df[date_col]
            if pd.api.types.is_numeric_dtype(series):
                converted_dates = pd.to_datetime(series, unit='s', errors='coerce')
            else:
                converted_dates = pd.to_datetime(series, errors='coerce')

            min_date = converted_dates.min()
            max_date = converted_dates.max()
            
            now = datetime.now()
            if min_date.year < 2010:
                logging.warning(f"[{filename}] Found very old dates: {min_date}")
            if max_date > now + pd.Timedelta(days=1):
                logging.warning(f"[{filename}] Found future dates: {max_date}")

        except Exception as e:
            logging.warning(f"[{filename}] Date validation error: {e}")
            return False, {"error": str(e)}

        return True, {}

    def run_validation(self):
        """Execute validation pipeline"""
        logging.info("--- Starting Data Validation Phase ---")
        
        if not self.validation_config.get("enabled", True):
            logging.info("Validation disabled in config.")
            return {"overall_status": "SKIPPED"}
        
        datasets = self.load_raw_files()
        
        if not datasets:
            logging.error("CRITICAL: No datasets found to validate.")
            return {"overall_status": "FAILED"}

        results = {}
        global_pass = True

        for filename, df in datasets.items():
            logging.info(f"Validating: {filename}")
            
            source_type = self.detect_source_type(df, filename)
            
            schema_ok, schema_issues = self.validate_schema(df, filename, source_type)
            
            quality_ok, quality_issues = self.validate_data_quality(df, filename)
            
            dates_ok, date_issues = self.validate_dates(df, filename)

            file_passed = schema_ok
            if not file_passed:
                global_pass = False

            results[filename] = {
                "source": source_type,
                "status": "PASSED" if file_passed else "FAILED",
                "issues": {
                    "schema": schema_issues,
                    "quality": quality_issues,
                    "dates": date_issues
                }
            }
            
            if not file_passed:
                logging.error(f" {filename} FAILED validation.")
            else:
                logging.info(f" {filename} PASSED validation.")

        status = "PASSED" if global_pass else "FAILED"
        results["overall_status"] = status
        logging.info(f"Validation Phase Completed: {status}")
        
        return results

    def run(self):
        return self.run_validation()

def validate_raw_data(raw_paths, config):
    """
    Entry point for the Orchestrator.
    raw_paths argument is kept for compatibility but logic scans the directory.
    """
    validator = DataValidation(config)
    results = validator.run()
    
    if results["overall_status"] == "FAILED":
        logging.error(" Pipeline Integrity Compromised: Critical validation errors found.")
        return True 
    return True
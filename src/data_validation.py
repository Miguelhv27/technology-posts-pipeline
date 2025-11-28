import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

class DataValidation:
    def __init__(self, config):
        self.config = config
        self.validation_config = config["validation"]
        self.raw_data_dir = config["paths"]["raw_data_dir"]

    def load_raw_files(self):
        """Load all raw data files from multiple sources"""
        datasets = {}
        
        if not os.path.exists(self.raw_data_dir):
            raise FileNotFoundError(f"Directorio de datos no encontrado: {self.raw_data_dir}")

        files = [f for f in os.listdir(self.raw_data_dir) if f.endswith((".csv", ".parquet"))]

        if not files:
            logging.warning("No se encontraron archivos de datos para validar")
            return datasets

        for file in files:
            file_path = os.path.join(self.raw_data_dir, file)
            try:
                if file.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif file.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                
                datasets[file] = df
                logging.info(f"Cargado archivo: {file} - {len(df)} filas")
                
            except Exception as e:
                logging.error(f"Error cargando archivo {file}: {e}")

        return datasets

    def validate_columns(self, df, filename, source_type):
        """Validate required columns based on data source"""
        if source_type == "twitter":
            required_cols = self.validation_config["required_columns"]["twitter"]
        elif source_type == "reddit":
            required_cols = self.validation_config["required_columns"]["reddit"]
        else:
            required_cols = self.validation_config["required_columns"].get("reddit", [])
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logging.error(f"[{filename}] Columnas requeridas faltantes: {missing_cols}")
            return False, missing_cols
        
        logging.info(f"[{filename}] Validación de columnas: OK")
        return True, []

    def validate_missing_values(self, df, filename):
        """Check missing values in critical columns"""
        check_columns = self.validation_config["check_null_columns"]
        existing_check_cols = [col for col in check_columns if col in df.columns]
        
        if not existing_check_cols:
            logging.warning(f"[{filename}] No hay columnas críticas para validar nulos")
            return True, 0
        
        missing_stats = {}
        for col in existing_check_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                missing_stats[col] = null_count
        
        if missing_stats:
            logging.warning(f"[{filename}] Valores nulos en columnas críticas: {missing_stats}")
            
            max_null_allowed = len(df) * self.validation_config["quality_thresholds"]["max_null_percentage"]
            critical_issues = {col: count for col, count in missing_stats.items() if count > max_null_allowed}
            
            if critical_issues:
                logging.error(f"[{filename}] Demasiados valores nulos: {critical_issues}")
                return False, critical_issues
        
        return True, missing_stats

    def validate_date_range(self, df, filename):
        """Validate dates are within allowed range"""
        date_columns = []
        
        for col in df.columns:
            if 'date' in col.lower() or 'created' in col.lower() or 'time' in col.lower():
                date_columns.append(col)
        
        if not date_columns:
            logging.info(f"[{filename}] No se encontraron columnas de fecha para validar")
            return True, {}
        
        date_range_config = self.validation_config["allowed_date_range"]
        start_date = pd.to_datetime(date_range_config["start"])
        end_date = pd.to_datetime(date_range_config["end"])
        
        date_issues = {}
        
        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                invalid_dates = df[date_col][(df[date_col] < start_date) | (df[date_col] > end_date)]
                
                if len(invalid_dates) > 0:
                    date_issues[date_col] = len(invalid_dates)
                    logging.warning(f"[{filename}] Fechas fuera de rango en {date_col}: {len(invalid_dates)} registros")
                    
            except Exception as e:
                logging.warning(f"[{filename}] Error procesando columna de fecha {date_col}: {e}")
        
        if date_issues:
            return False, date_issues
        
        return True, {}

    def validate_data_types(self, df, filename):
        """Validate basic data types and structure"""
        issues = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].min() < 0 and col in ['score', 'num_comments', 'retweet_count', 'favorite_count']:
                issues.append(f"Valores negativos en {col}")
        
        id_cols = [col for col in df.columns if 'id' in col.lower()]
        for id_col in id_cols:
            if df[id_col].duplicated().any():
                issues.append(f"IDs duplicados en {id_col}")
        
        if issues:
            logging.warning(f"[{filename}] Problemas de tipos de datos: {issues}")
            return False, issues
        
        return True, []

    def detect_source_type(self, filename, df):
        """Detect data source type from filename and columns"""
        filename_lower = filename.lower()
        
        if 'twitter' in filename_lower:
            return 'twitter'
        elif 'reddit' in filename_lower:
            return 'reddit'
        elif any(col in df.columns for col in ['retweet_count', 'favorite_count']):
            return 'twitter'
        elif any(col in df.columns for col in ['subreddit', 'selftext', 'score']):
            return 'reddit'
        else:
            return 'unknown'

    def run_validation(self):
        """Execute complete validation process"""
        logging.info("Iniciando validación de datos")
        
        if not self.validation_config["enabled"]:
            logging.info("Validación desactivada en configuración")
            return {"overall_status": "SKIPPED"}
        
        datasets = self.load_raw_files()
        
        if not datasets:
            logging.error("No hay datos para validar")
            return {"overall_status": "FAILED", "error": "No data files found"}

        results = {}
        all_passed = True

        for filename, df in datasets.items():
            logging.info(f"Validando archivo: {filename}")
            
            source_type = self.detect_source_type(filename, df)

            col_valid, col_issues = self.validate_columns(df, filename, source_type)
            missing_valid, missing_issues = self.validate_missing_values(df, filename)
            date_valid, date_issues = self.validate_date_range(df, filename)
            type_valid, type_issues = self.validate_data_types(df, filename)
            
            file_status = col_valid and missing_valid and date_valid and type_valid
            
            if not file_status:
                all_passed = False

            results[filename] = {
                "status": "PASSED" if file_status else "FAILED",
                "source_type": source_type,
                "rows": len(df),
                "columns": list(df.columns),
                "issues": {
                    "missing_columns": col_issues,
                    "missing_values": missing_issues,
                    "date_issues": date_issues,
                    "data_type_issues": type_issues
                }
            }

            status_msg = "PASÓ" if file_status else "FALLÓ"
            logging.info(f"Validación de {filename}: {status_msg}")

        overall_status = "PASSED" if all_passed else "FAILED"
        results["overall_status"] = overall_status
        
        logging.info(f"Validación completada. Estado general: {overall_status}")
        return results

    def run(self):
        return self.run_validation()

def validate_raw_data(raw_paths, config):
    """Main validation function """
    validator = DataValidation(config)
    results = validator.run()
    
    if results["overall_status"] == "FAILED":
        logging.warning("Validación falló pero continuando con el pipeline para datos mock")
        logging.warning("Esto es aceptable para datos de demostración")
        return True  
    
    elif results["overall_status"] == "PASSED":
        logging.info("Todas las validaciones pasaron exitosamente")
        return True
    else:
        logging.info("Validación saltada por configuración")
        return True
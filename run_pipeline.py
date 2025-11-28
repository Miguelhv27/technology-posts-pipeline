#!/usr/bin/env python3
"""
Main execution script for Technology Posts Pipeline
Orchestrates the complete DataOps pipeline for social media analysis
"""

import os
import sys
import logging
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator import Orchestrator
from src.data_ingestion import ingest_data
from src.data_validation import validate_raw_data
from src.data_transformation import transform_data
from src.nlp import run_nlp_pipeline
from src.network_analysis import run_network_analysis
from src.analysis import run_statistical_analysis

def setup_logging(log_dir="logs/"):
    """Configure comprehensive logging for the pipeline"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return log_file

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Technology Posts Data Pipeline')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/pipeline_config.yaml',
        help='Path to pipeline configuration file'
    )
    
    parser.add_argument(
        '--skip-ingestion', 
        action='store_true',
        help='Skip data ingestion step'
    )
    
    parser.add_argument(
        '--skip-validation', 
        action='store_true',
        help='Skip data validation step'
    )
    
    parser.add_argument(
        '--skip-transformation', 
        action='store_true', 
        help='Skip data transformation step'
    )
    
    parser.add_argument(
        '--skip-nlp', 
        action='store_true',
        help='Skip NLP processing step'
    )
    
    parser.add_argument(
        '--skip-network', 
        action='store_true',
        help='Skip network analysis step'
    )
    
    parser.add_argument(
        '--skip-analysis', 
        action='store_true',
        help='Skip statistical analysis step'
    )
    
    parser.add_argument(
        '--steps', 
        type=str,
        help='Comma-separated list of specific steps to run (ingestion,validation,transformation,nlp,network,analysis)'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Validate configuration without executing pipeline'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        help='Custom output directory for results'
    )
    
    return parser.parse_args()

def validate_environment():
    """Validate that required environment variables are set"""
    required_vars = [
        'TWITTER_API_KEY',
        'TWITTER_API_SECRET', 
        'TWITTER_BEARER_TOKEN',
        'REDDIT_CLIENT_ID',
        'REDDIT_CLIENT_SECRET'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logging.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logging.warning("Some data sources may not work without proper API credentials")
        return False
    
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'pandas',
        'numpy', 
        'scipy',
        'sklearn', 
        'statsmodels',
        'networkx',
        'matplotlib',
        'nltk',
        'textblob',
        'vaderSentiment',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logging.warning(f"Missing packages (but continuing): {', '.join(missing_packages)}")
        return True
    
    return True

def run_individual_steps(config, args):
    """Run pipeline steps individually with manual orchestration"""
    logging.info("Running pipeline steps individually...")
    
    results = {}
    
    try:
        if not args.skip_ingestion and (not args.steps or 'ingestion' in args.steps):
            logging.info("=" * 60)
            logging.info("STEP 1: DATA INGESTION")
            logging.info("=" * 60)
            raw_paths = ingest_data(config)
            results['ingestion'] = raw_paths
        else:
            logging.info("Skipping data ingestion")
            results['ingestion'] = None
         
        if not args.skip_validation and (not args.steps or 'validation' in args.steps):
            logging.info("=" * 60)
            logging.info("STEP 2: DATA VALIDATION")
            logging.info("=" * 60)
            if results.get('ingestion'):
                validation_passed = validate_raw_data(results['ingestion'], config)
                results['validation'] = validation_passed
                if not validation_passed:
                    logging.error("Data validation failed. Stopping pipeline.")
                    return results
            else:
                logging.warning("No ingestion results available for validation")
        else:
            logging.info("Skipping data validation")
            results['validation'] = True
        
        if not args.skip_transformation and (not args.steps or 'transformation' in args.steps):
            logging.info("=" * 60)
            logging.info("STEP 3: DATA TRANSFORMATION")
            logging.info("=" * 60)
            if results.get('ingestion'):
                transformed_data = transform_data(results['ingestion'], config)
                results['transformation'] = transformed_data
            else:
                logging.error("No ingestion results available for transformation")
                return results
        else:
            logging.info("Skipping data transformation")
            results['transformation'] = None
        
        if not args.skip_nlp and (not args.steps or 'nlp' in args.steps):
            logging.info("=" * 60)
            logging.info("STEP 4: NLP PROCESSING")
            logging.info("=" * 60)
            if results.get('transformation') is not None:
                nlp_processed_data, nlp_results = run_nlp_pipeline(results['transformation'], config)
                results['nlp'] = {
                    'data': nlp_processed_data,
                    'results': nlp_results
                }
            else:
                logging.error("No transformation results available for NLP")
                return results
        else:
            logging.info("Skipping NLP processing")
            results['nlp'] = {'data': results.get('transformation'), 'results': {}}
        
        if not args.skip_network and (not args.steps or 'network' in args.steps):
            logging.info("=" * 60)
            logging.info("STEP 5: NETWORK ANALYSIS")
            logging.info("=" * 60)
            if results['nlp']['data'] is not None:
                network_results = run_network_analysis(results['nlp']['data'], config)
                results['network'] = network_results
            else:
                logging.error("No NLP results available for network analysis")
                return results
        else:
            logging.info("Skipping network analysis")
            results['network'] = {}
        
        if not args.skip_analysis and (not args.steps or 'analysis' in args.steps):
            logging.info("=" * 60)
            logging.info("STEP 6: STATISTICAL ANALYSIS")
            logging.info("=" * 60)
            if results['nlp']['data'] is not None:
                analysis_results = run_statistical_analysis(
                    results['nlp']['data'], 
                    results.get('network', {}),
                    config
                )
                results['analysis'] = analysis_results
            else:
                logging.error("No data available for statistical analysis")
                return results
        else:
            logging.info("Skipping statistical analysis")
            results['analysis'] = {}
        
        logging.info("=" * 60)
        logging.info("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY")
        logging.info("=" * 60)
        
        return results
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise

def generate_summary_report(results, log_file):
    """Generate a summary report of the pipeline execution"""
    logging.info("=" * 60)
    logging.info("PIPELINE EXECUTION SUMMARY")
    logging.info("=" * 60)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'log_file': log_file,
        'steps_completed': []
    }
    
    if results.get('ingestion'):
        summary['steps_completed'].append('ingestion')
        summary['ingestion_sources'] = list(results['ingestion'].keys()) if isinstance(results['ingestion'], dict) else ['unknown']
    
    if results.get('validation'):
        summary['steps_completed'].append('validation')
        summary['validation_passed'] = results['validation']
    
    if results.get('transformation') is not None:
        summary['steps_completed'].append('transformation')
        summary['transformed_records'] = len(results['transformation']) if hasattr(results['transformation'], '__len__') else 'unknown'
    
    if results.get('nlp', {}).get('results'):
        summary['steps_completed'].append('nlp')
        summary['nlp_topics'] = len(results['nlp']['results'].get('topics', []))
    
    if results.get('network'):
        summary['steps_completed'].append('network')
        summary['influencers_identified'] = len(results['network'].get('twitter_influencers', {})) + len(results['network'].get('reddit_influencers', {}))
    
    if results.get('analysis'):
        summary['steps_completed'].append('analysis')
        summary['correlations_calculated'] = len(results['analysis'].get('correlations', {}))
    
    for key, value in summary.items():
        if key != 'timestamp': 
            logging.info(f"{key.replace('_', ' ').title()}: {value}")
    
    return summary

def main():
    """Main pipeline execution function"""
    args = parse_arguments()
    
    log_file = setup_logging()
    logging.info("Technology Posts Pipeline - Starting Execution")
    logging.info(f"Configuration file: {args.config}")
    logging.info(f"Log file: {log_file}")
    
    try:
        if not validate_environment():
            logging.warning("Environment validation failed - some features may not work")
        
        if not check_dependencies():
            logging.error("Dependency check failed - exiting")
            return 1
        
        if not os.path.exists(args.config):
            logging.error(f"Configuration file not found: {args.config}")
            return 1
        
        orchestrator = Orchestrator(args.config)
        
        if args.dry_run:
            logging.info("DRY RUN MODE - Configuration validated successfully")
            logging.info("Pipeline would execute the following steps:")
            steps_to_run = []
            if not args.skip_ingestion: steps_to_run.append("Ingestion")
            if not args.skip_validation: steps_to_run.append("Validation") 
            if not args.skip_transformation: steps_to_run.append("Transformation")
            if not args.skip_nlp: steps_to_run.append("NLP")
            if not args.skip_network: steps_to_run.append("Network Analysis")
            if not args.skip_analysis: steps_to_run.append("Statistical Analysis")
            
            logging.info(f"Steps: {', '.join(steps_to_run)}")
            return 0
        
        if args.steps or any([args.skip_ingestion, args.skip_validation, args.skip_transformation, 
                            args.skip_nlp, args.skip_network, args.skip_analysis]):
            results = run_individual_steps(orchestrator.config, args)
        else:
            logging.info("Running complete pipeline using orchestrator...")
            results = orchestrator.run()
        
        summary = generate_summary_report(results, log_file)
        
        logging.info("Pipeline execution completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"Pipeline execution failed with error: {e}")
        logging.error("Check the log file for detailed error information")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
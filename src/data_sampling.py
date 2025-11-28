import pandas as pd
import logging

def sample_data_for_development(df, sample_fraction=0.3, min_samples=5000): 
    """
    Sample data for development/testing to reduce computational load
    UPDATED FOR REAL DATA: Keep more data for meaningful analysis
    """
    logging.info(f"Original data size: {len(df)} records")
    
    if len(df) <= 10000:  
        sample_fraction = 0.8
    
    if len(df) <= min_samples:
        logging.info(f"Data already at desired size ({len(df)} records), skipping sampling")
        return df

    try:
        if 'data_source' in df.columns:
            sampled_data = df.groupby('data_source', group_keys=False).apply(
                lambda x: x.sample(frac=sample_fraction, random_state=42)
            )
        else:
            sampled_data = df.sample(frac=sample_fraction, random_state=42)

        if len(sampled_data) < min_samples:
            logging.info(f"Sampled data too small ({len(sampled_data)}), taking {min_samples} records")
            sampled_data = df.sample(n=min_samples, random_state=42)
            
        logging.info(f"Sampled data size: {len(sampled_data)} records")
        return sampled_data.reset_index(drop=True)
        
    except Exception as e:
        logging.warning(f"Sampling failed, using original data: {e}")
        return df
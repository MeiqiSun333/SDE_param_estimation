"""
Data preprocessing module.

Handles loading, cleaning, and transforming longitudinal data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional


class EmotionDataProcessor:
    """
    Process longitudinal emotion survey data.
    
    Attributes:
        skip_cols: Columns to exclude from data cleaning
        emotion_vars: Target variables for modeling
    """
    
    def __init__(self, skip_cols: Optional[List[str]] = None):
        """
        Initialize data processor.
        
        Args:
            skip_cols: Columns to skip during cleaning (default: UUID, Date, Time, dataset)
        """
        self.skip_cols = skip_cols or ['UUID', 'Date_Local', 'Time_Local', 'dataset']
        self.emotion_vars = ['SAD_ES', 'STR_ES', 'SITMOD_ES', 'DIST_ES', 
                            'REAP1_ES', 'RUM_ES', 'EMOCOPE_ES']
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw data from CSV.
        
        Returns:
            Raw dataframe
        """

        df = pd.read_csv(filepath)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with all missing values.
        
        Returns:
            Cleaned dataframe
        """

        data_cols = [col for col in df.columns if col not in self.skip_cols]

        df_clean = df[
            df[data_cols].replace('', np.nan).replace(0, np.nan).notna().any(axis=1)
        ].copy()

        return df_clean
    
    def engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create datetime and relative time features.
        
        Args:
            df: Dataframe with Date_Local and Time_Local columns
            
        Returns:
            Dataframe with added datetime and time_hours columns
        """
        
        # Combine date and time
        df['datetime'] = pd.to_datetime(
            df['Date_Local'].astype(str) + ' ' + df['Time_Local'].astype(str),
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'
        )
        
        df = df.sort_values(['UUID', 'datetime'])
        
        # Calculate hours since first observation per subject
        df['time_hours'] = df.groupby('UUID')['datetime'].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 3600
        )
        
        return df
    
    def get_descriptive_stats(self, df: pd.DataFrame) -> dict:
        """
        Calculate descriptive statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'n_subjects': df['UUID'].nunique(),
            'n_observations': len(df),
            'obs_per_subject': df.groupby('UUID').size().describe().to_dict(),
            'emotion_stats': df[self.emotion_vars].describe().to_dict()
        }
        
        return stats
    
    def prepare_modeling_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final dataset.
        
        Returns:
            Dataframe with renamed columns
        """
        
        # Select relevant columns
        model_cols = ["UUID", "SAD_ES", "STR_ES", "SITMOD_ES", "DIST_ES", 
                     "REAP1_ES", "RUM_ES", "EMOCOPE_ES", "time_hours"]
        
        df_model = df[model_cols].copy()
        
        # Remove rows with missing values in emotion variables
        emotion_cols = [col for col in model_cols if col not in ["UUID", "time_hours"]]
        df_model = df_model[
            df_model[emotion_cols].replace('', np.nan).replace(0, np.nan).notna().any(axis=1)
        ].copy()
        
        # Rename columns for clarity
        df_model.columns = ["UUID", "SAD", "STR", "SITMOD", "DIST", 
                           "REAP", "RUM", "EMOCOPE", "time_hours"]
        
        
        return df_model
    
    def process_pipeline(self, input_path: str, output_path: Optional[str] = None):
        """
        Run full preprocessing pipeline.
        
        Args:
            input_path: Path to raw CSV file
            output_path: Optional path to save processed data
            
        Returns:
            Tuple of (processed_dataframe, statistics_dict)
        """

        df = self.load_data(input_path)
        df = self.clean_data(df)
        
        # Engineer time features
        df = self.engineer_time_features(df)
        df_model = self.prepare_modeling_data(df)
        
        # Get statistics
        stats = self.get_descriptive_stats(df)
        
        # Save if output path provided
        if output_path:
            df_model.to_csv(output_path, index=False)
        
        return df_model, stats


def main():
    
    # Initialize processor
    processor = EmotionDataProcessor()
    
    # Define paths
    raw_data_path = "data/raw/data.csv"
    processed_data_path = "data/processed/emotion_data_clean.csv"
    
    # Run pipeline
    df_model, stats = processor.process_pipeline(raw_data_path, processed_data_path)
    

    print("Data processing complete!")
    print(f"Subjects: {stats['n_subjects']}")
    print(f"Observations: {stats['n_observations']}")
    print(f"\nFirst few rows:")
    print(df_model.head())


if __name__ == "__main__":
    main()

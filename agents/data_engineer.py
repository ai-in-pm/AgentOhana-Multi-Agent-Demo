import pandas as pd
import numpy as np
from typing import Dict, List

class DataEngineer:
    def __init__(self):
        self.data_sources = [
            "navigation_trajectories",
            "dialogue_trajectories",
            "coding_trajectories"
        ]
        
    def process_data(self) -> Dict[str, pd.DataFrame]:
        """Process and standardize data from multiple sources."""
        processed_data = {}
        
        # Simulate loading and processing data from different sources
        for source in self.data_sources:
            # Generate synthetic data for demonstration
            n_samples = 3000  # 3000 samples per source
            trajectories = self._generate_synthetic_data(source, n_samples)
            processed_data[source] = self._standardize_data(trajectories, source)
            
        return processed_data
    
    def _generate_synthetic_data(self, source: str, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data for demonstration purposes."""
        if source == "navigation_trajectories":
            data = {
                'action': np.random.choice(['move_forward', 'turn_left', 'turn_right', 'stop'], n_samples),
                'state': [f"position_{i}" for i in range(n_samples)],
                'reward': np.random.normal(0, 1, n_samples),
                'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='S')
            }
        elif source == "dialogue_trajectories":
            data = {
                'user_input': [f"user_message_{i}" for i in range(n_samples)],
                'agent_response': [f"agent_response_{i}" for i in range(n_samples)],
                'sentiment': np.random.choice(['positive', 'neutral', 'negative'], n_samples),
                'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='S')
            }
        else:  # coding_trajectories
            data = {
                'code_input': [f"code_snippet_{i}" for i in range(n_samples)],
                'agent_suggestion': [f"suggestion_{i}" for i in range(n_samples)],
                'acceptance': np.random.choice([True, False], n_samples),
                'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='S')
            }
        
        return pd.DataFrame(data)
    
    def _standardize_data(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Convert source-specific data into AgentOhana's unified format."""
        standardized = pd.DataFrame()
        
        # Common fields across all trajectories
        standardized['timestamp'] = df['timestamp']
        standardized['source'] = source
        
        # Source-specific standardization
        if source == "navigation_trajectories":
            standardized['input'] = df['state']
            standardized['output'] = df['action']
            standardized['metadata'] = df['reward'].apply(lambda x: {'reward': float(x)})
        elif source == "dialogue_trajectories":
            standardized['input'] = df['user_input']
            standardized['output'] = df['agent_response']
            standardized['metadata'] = df['sentiment'].apply(lambda x: {'sentiment': x})
        else:  # coding_trajectories
            standardized['input'] = df['code_input']
            standardized['output'] = df['agent_suggestion']
            standardized['metadata'] = df['acceptance'].apply(lambda x: {'accepted': bool(x)})
        
        return standardized

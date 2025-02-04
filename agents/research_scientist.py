import pandas as pd
import numpy as np
from typing import Dict, Any

class ResearchScientist:
    def __init__(self):
        self.metrics = [
            "data_distribution",
            "sequence_length",
            "task_complexity"
        ]
    
    def analyze_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze the processed data and provide insights for training."""
        analysis = {}
        
        # Analyze data distribution
        distribution = self._analyze_distribution(data)
        analysis['distribution'] = distribution
        
        # Analyze sequence lengths
        seq_stats = self._analyze_sequence_lengths(data)
        analysis['sequence_stats'] = seq_stats
        
        # Generate training recommendations
        recommendations = self._generate_recommendations(distribution, seq_stats)
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def _analyze_distribution(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze the distribution of data across different sources."""
        total_samples = sum(len(df) for df in data.values())
        distribution = {
            source: len(df) / total_samples 
            for source, df in data.items()
        }
        return distribution
    
    def _analyze_sequence_lengths(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Analyze the sequence length statistics for each source."""
        seq_stats = {}
        for source, df in data.items():
            # Group by some identifier (could be conversation_id, trajectory_id, etc.)
            # For demo, we'll use timestamp binned by minute as a proxy
            sequences = df.groupby(df['timestamp'].dt.floor('min'))
            
            stats = {
                'mean_length': sequences.size().mean(),
                'max_length': sequences.size().max(),
                'min_length': sequences.size().min()
            }
            seq_stats[source] = stats
        
        return seq_stats
    
    def _generate_recommendations(self, 
                                distribution: Dict[str, float],
                                seq_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate training recommendations based on analysis."""
        # Calculate maximum sequence length across all sources
        max_seq_length = max(
            stats['max_length'] 
            for stats in seq_stats.values()
        )
        
        # Calculate average complexity (placeholder metric)
        avg_complexity = np.mean([
            stats['mean_length'] / stats['max_length']
            for stats in seq_stats.values()
        ])
        
        # Generate recommendations
        recommendations = {
            'learning_rate': self._recommend_learning_rate(avg_complexity),
            'batch_size': self._recommend_batch_size(distribution),
            'sequence_length': int(max_seq_length * 1.2),  # Add 20% buffer
            'training_steps': self._recommend_training_steps(distribution)
        }
        
        return recommendations
    
    def _recommend_learning_rate(self, complexity: float) -> float:
        """Recommend learning rate based on task complexity."""
        # Lower learning rate for more complex tasks
        base_lr = 1e-4
        return base_lr * (1 - complexity * 0.5)
    
    def _recommend_batch_size(self, distribution: Dict[str, float]) -> int:
        """Recommend batch size based on data distribution."""
        # Smaller batch size if data is highly imbalanced
        max_imbalance = max(distribution.values()) - min(distribution.values())
        base_batch = 32
        return max(8, int(base_batch * (1 - max_imbalance)))
    
    def _recommend_training_steps(self, distribution: Dict[str, float]) -> int:
        """Recommend number of training steps."""
        # More steps if data is diverse
        num_sources = len(distribution)
        base_steps = 10000
        return int(base_steps * (1 + 0.2 * num_sources))

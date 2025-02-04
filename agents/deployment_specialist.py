from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any
import json
import os

class DeploymentSpecialist:
    def __init__(self):
        self.model_dir = "deployed_model"
        self.config_file = "config.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def deploy_model(self, model: AutoModelForCausalLM) -> Dict[str, Any]:
        """Deploy the trained model and return endpoint information."""
        # Save model and configuration
        model_info = self._save_model(model)
        
        # Set up serving configuration
        serving_config = self._setup_serving(model_info)
        
        # Create deployment endpoint
        endpoint = self._create_endpoint(serving_config)
        
        return endpoint
    
    def _save_model(self, model: AutoModelForCausalLM) -> Dict[str, str]:
        """Save the model and its configuration."""
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(self.model_dir, "model")
        model.save_pretrained(model_path)
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(model_path)
        
        # Save configuration
        config = {
            'model_name': 'AgentOhana-xLAM',
            'base_model': 'xLAM-v0.1',
            'model_path': model_path,
            'device': str(self.device)
        }
        
        config_path = os.path.join(self.model_dir, self.config_file)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def _setup_serving(self, model_info: Dict[str, str]) -> Dict[str, Any]:
        """Set up the serving configuration."""
        serving_config = {
            'service_name': 'agent-ohana-service',
            'model_config': model_info,
            'endpoints': {
                'predict': '/api/v1/predict',
                'health': '/api/v1/health'
            },
            'scaling': {
                'min_replicas': 1,
                'max_replicas': 3,
                'target_cpu_utilization': 80
            }
        }
        
        # Save serving configuration
        config_path = os.path.join(self.model_dir, 'serving_config.json')
        with open(config_path, 'w') as f:
            json.dump(serving_config, f, indent=2)
        
        return serving_config
    
    def _create_endpoint(self, serving_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create and return the deployment endpoint information."""
        # For demo purposes, we'll simulate a local endpoint
        endpoint = {
            'service_url': 'http://localhost:8000',
            'endpoints': serving_config['endpoints'],
            'model_info': {
                'name': serving_config['model_config']['model_name'],
                'version': '1.0.0'
            },
            'status': 'active',
            'documentation': {
                'description': 'AgentOhana xLAM API',
                'example_request': {
                    'method': 'POST',
                    'endpoint': '/api/v1/predict',
                    'body': {
                        'input': 'Your input text here',
                        'max_length': 100
                    }
                }
            }
        }
        
        # Save endpoint information
        endpoint_path = os.path.join(self.model_dir, 'endpoint_info.json')
        with open(endpoint_path, 'w') as f:
            json.dump(endpoint, f, indent=2)
        
        return endpoint

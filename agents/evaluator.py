from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from typing import Dict, Any, List

class Evaluator:
    def __init__(self):
        self.benchmarks = {
            'webshop': self._evaluate_webshop,
            'hotpotqa': self._evaluate_hotpotqa,
            'coding': self._evaluate_coding
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        
    def evaluate_model(self, model: AutoModelForCausalLM) -> Dict[str, Any]:
        """Evaluate the model on multiple benchmarks."""
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
        results = {}
        
        # Run each benchmark
        for benchmark_name, benchmark_fn in self.benchmarks.items():
            results[benchmark_name] = benchmark_fn(model)
        
        # Calculate aggregate metrics
        results['aggregate'] = self._calculate_aggregate_metrics(results)
        
        return results
    
    def _evaluate_webshop(self, model: AutoModelForCausalLM) -> Dict[str, float]:
        """Evaluate model on WebShop e-commerce assistant tasks."""
        # Sample test cases
        test_cases = [
            {
                'input': 'Find a blue cotton t-shirt under $30',
                'expected_actions': ['search', 'filter_color', 'filter_material', 'filter_price']
            },
            {
                'input': 'Add the cheapest item to cart',
                'expected_actions': ['sort_price_asc', 'select_first', 'add_to_cart']
            }
        ]
        
        correct = 0
        total_actions = 0
        
        for case in test_cases:
            # Generate model response
            input_text = f"Task: {case['input']}\nActions:"
            inputs = self.tokenizer(input_text, 
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=100,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and evaluate actions
            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_actions = self._extract_actions(predicted_text)
            
            # Count correct actions
            for action in predicted_actions:
                if action in case['expected_actions']:
                    correct += 1
                total_actions += 1
        
        return {
            'accuracy': correct / total_actions if total_actions > 0 else 0,
            'total_cases': len(test_cases)
        }
    
    def _evaluate_hotpotqa(self, model: AutoModelForCausalLM) -> Dict[str, float]:
        """Evaluate model on HotpotQA multi-hop reasoning tasks."""
        # Sample test cases
        test_cases = [
            {
                'question': 'Who wrote the book that inspired the movie starring Tom Hanks as a FedEx executive?',
                'context': [
                    'Cast Away is a 2000 American survival drama film directed by Robert Zemeckis and starring Tom Hanks as a FedEx executive.',
                    'The film was inspired by the book "Lost at Sea" by Richard Maxwell.'
                ],
                'answer': 'Richard Maxwell'
            }
        ]
        
        correct = 0
        
        for case in test_cases:
            # Format input
            input_text = f"Question: {case['question']}\nContext: {' '.join(case['context'])}\nAnswer:"
            inputs = self.tokenizer(input_text, 
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True).to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=50,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            predicted_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check exact match
            if case['answer'].lower() in predicted_answer.lower():
                correct += 1
        
        return {
            'accuracy': correct / len(test_cases),
            'total_cases': len(test_cases)
        }
    
    def _evaluate_coding(self, model: AutoModelForCausalLM) -> Dict[str, float]:
        """Evaluate model on coding assistance tasks."""
        test_cases = [
            {
                'prompt': 'Write a function to calculate fibonacci numbers',
                'expected_elements': ['def', 'fibonacci', 'return']
            }
        ]
        
        correct_elements = 0
        total_elements = 0
        
        for case in test_cases:
            input_text = f"Task: {case['prompt']}\nCode:"
            inputs = self.tokenizer(input_text, 
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=200,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check for expected elements
            for element in case['expected_elements']:
                if element in generated_code:
                    correct_elements += 1
                total_elements += 1
        
        return {
            'element_accuracy': correct_elements / total_elements if total_elements > 0 else 0,
            'total_cases': len(test_cases)
        }
    
    def _calculate_aggregate_metrics(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate aggregate metrics across all benchmarks."""
        # Extract accuracies
        accuracies = []
        if 'accuracy' in results['webshop']:
            accuracies.append(results['webshop']['accuracy'])
        if 'accuracy' in results['hotpotqa']:
            accuracies.append(results['hotpotqa']['accuracy'])
        if 'element_accuracy' in results['coding']:
            accuracies.append(results['coding']['element_accuracy'])
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies)
        }
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract actions from generated text."""
        # Simple extraction - split by newlines and clean
        actions = []
        for line in text.split('\n'):
            if ':' in line:
                action = line.split(':')[1].strip().lower()
                actions.append(action)
        return actions

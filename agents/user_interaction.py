import requests
from typing import Dict, Any
import json

class UserInteractionFacilitator:
    def __init__(self):
        self.response_templates = {
            'greeting': "Hello! I'm the AgentOhana assistant. How can I help you today?",
            'farewell': "Thank you for using AgentOhana! Have a great day!",
            'error': "I apologize, but I encountered an error. Please try again or rephrase your request.",
            'clarification': "Could you please provide more details about your request?"
        }
    
    def handle_user_input(self, user_input: str, endpoint: Dict[str, Any]) -> str:
        """Handle user input and return appropriate response."""
        try:
            # Clean and validate input
            cleaned_input = self._clean_input(user_input)
            
            # Get response from model
            response = self._get_model_response(cleaned_input, endpoint)
            
            # Format and enhance response
            formatted_response = self._format_response(response)
            
            return formatted_response
            
        except Exception as e:
            return f"{self.response_templates['error']} (Error: {str(e)})"
    
    def _clean_input(self, user_input: str) -> str:
        """Clean and validate user input."""
        # Remove extra whitespace
        cleaned = " ".join(user_input.split())
        
        # Basic input validation
        if not cleaned:
            raise ValueError("Empty input")
        
        return cleaned
    
    def _get_model_response(self, user_input: str, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Get response from the deployed model."""
        try:
            # Prepare request
            url = f"{endpoint['service_url']}{endpoint['endpoints']['predict']}"
            payload = {
                'input': user_input,
                'max_length': 100
            }
            
            # For demo, simulate API call
            response = self._simulate_model_response(user_input)
            
            return response
            
        except Exception as e:
            raise ConnectionError(f"Failed to get model response: {str(e)}")
    
    def _simulate_model_response(self, user_input: str) -> Dict[str, Any]:
        """Simulate model response for demonstration."""
        # Simple response generation based on input keywords
        if 'help' in user_input.lower() or 'what' in user_input.lower():
            return {
                'response': "AgentOhana is a unified framework for training AI agents. It combines data from various environments to create more versatile and capable agents.",
                'confidence': 0.95
            }
        elif 'how' in user_input.lower():
            return {
                'response': "I can help you with tasks like planning, answering questions, or explaining concepts. What specific task would you like help with?",
                'confidence': 0.90
            }
        elif 'plan' in user_input.lower():
            return {
                'response': "I'll help you create a plan. Let's break this down into steps:\n1. First, let's identify your goal\n2. Then, we'll outline the necessary steps\n3. Finally, we'll set a timeline",
                'confidence': 0.85
            }
        else:
            return {
                'response': "I understand your request. Let me help you with that. Could you provide more specific details about what you'd like to achieve?",
                'confidence': 0.75
            }
    
    def _format_response(self, response: Dict[str, Any]) -> str:
        """Format the model response for user presentation."""
        # Extract main response
        main_response = response.get('response', self.response_templates['error'])
        
        # Add confidence indicator if available
        confidence = response.get('confidence', 0.0)
        confidence_indicator = "🎯" if confidence > 0.9 else "✨" if confidence > 0.7 else "💭"
        
        # Format final response
        formatted = f"{confidence_indicator} {main_response}"
        
        return formatted

import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from agents.data_engineer import DataEngineer
from agents.research_scientist import ResearchScientist
from agents.model_trainer import ModelTrainer
from agents.evaluator import Evaluator
from agents.deployment_specialist import DeploymentSpecialist
from agents.user_interaction import UserInteractionFacilitator
import sys
import io

# Load environment variables
load_dotenv()

class AgentOhanaDemo:
    def __init__(self):
        self.data_engineer = DataEngineer()
        self.research_scientist = ResearchScientist()
        self.model_trainer = ModelTrainer()
        self.evaluator = Evaluator()
        self.deployment_specialist = DeploymentSpecialist()
        self.facilitator = UserInteractionFacilitator()
        
        # Initialize API keys from environment variables
        self.api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'groq': os.getenv('GROQ_API_KEY'),
            'google': os.getenv('GOOGLE_API_KEY'),
            'cohere': os.getenv('COHERE_API_KEY'),
            'emergenceai': os.getenv('EMERGENCEAI_API_KEY')
        }
        
        # Verify all API keys are present
        for key, value in self.api_keys.items():
            if not value:
                raise ValueError(f"Missing {key} API key in environment variables")

    def run_demo(self):
        st.title("AgentOhana Multi-Agent Demo 🌺")
        st.write("Welcome to the AgentOhana demonstration! Watch as our team of AI agents collaborates in real-time.")
        
        # Data Engineer Stage
        with st.expander("1. Data Engineer 📊", expanded=True):
            st.write("Data Engineer is processing the trajectory data...")
            data = self.data_engineer.process_data()
            st.success("✅ Data processing complete!")
            
        # Research Scientist Stage
        with st.expander("2. Research Scientist 🔬", expanded=True):
            st.write("Research Scientist is analyzing the data...")
            analysis = self.research_scientist.analyze_data(data)
            st.success("✅ Data analysis complete!")
            
        # Model Trainer Stage
        with st.expander("3. Model Trainer 🤖", expanded=True):
            st.write("Model Trainer is fine-tuning xLAM-v0.1...")
            
            # Create placeholder for training progress
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Capture training output
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                model = self.model_trainer.train_model(data, analysis)
                
                # Process captured output
                output_lines = captured_output.getvalue().split('\n')
                total_steps = 300  # 3 epochs * 100 steps
                current_step = 0
                
                for line in output_lines:
                    if "Step" in line and "Loss" in line:
                        current_step += 1
                        progress = current_step / total_steps
                        progress_text.text(line)
                        progress_bar.progress(progress)
                
                st.success("✅ Model training complete!")
                
            finally:
                sys.stdout = old_stdout
            
        # Evaluator Stage
        with st.expander("4. Evaluator 📈", expanded=True):
            st.write("Evaluator is testing the model...")
            results = self.evaluator.evaluate_model(model)
            st.success("✅ Model evaluation complete!")
            
        # Deployment Specialist Stage
        with st.expander("5. Deployment Specialist 🚀", expanded=True):
            st.write("Deployment Specialist is deploying the model...")
            endpoint = self.deployment_specialist.deploy_model(model)
            st.success("✅ Model deployment complete!")
            
        # User Interaction Stage
        st.markdown("---")
        st.header("Chat with AgentOhana 💬")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What would you like to know about AgentOhana?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = self.facilitator.handle_user_input(prompt, endpoint)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    demo = AgentOhanaDemo()
    demo.run_demo()

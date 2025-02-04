# AgentOhana Multi-Agent Demo 🌺

This project demonstrates a collaborative AI system using the AgentOhana framework, featuring six specialized AI agents working together to train and deploy an AI model.

## Overview

AgentOhana is a unified framework for training AI agents by aggregating and standardizing trajectories from multiple environments. This demo showcases how different AI agents collaborate in real-time to:

1. Process and standardize data (Data Engineer)
2. Analyze data and provide training insights (Research Scientist)
3. Fine-tune an xLAM model (Model Trainer)
4. Evaluate model performance (Evaluator)
5. Deploy the model (Deployment Specialist)
6. Interact with users (User Interaction Facilitator)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/agent-ohana-demo.git
cd agent-ohana-demo
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
COHERE_API_KEY=your_cohere_key
EMERGENCEAI_API_KEY=your_emergenceai_key
```

## Running the Demo

Start the Streamlit app:
```bash
streamlit run agent_ohana_demo.py
```

The demo will open in your default web browser, showing the real-time collaboration between agents and allowing you to interact with the system.

## Project Structure

```
agent-ohana-demo/
├── .env                    # API keys and configuration
├── requirements.txt        # Python dependencies
├── agent_ohana_demo.py    # Main demo script
├── agents/                 # Agent modules
│   ├── __init__.py
│   ├── data_engineer.py
│   ├── research_scientist.py
│   ├── model_trainer.py
│   ├── evaluator.py
│   ├── deployment_specialist.py
│   └── user_interaction.py
└── deployed_model/        # Directory for deployed model files
```

## Agent Roles

1. **Data Engineer**
   - Collects and processes trajectory data from various sources
   - Standardizes data into a unified format
   - Ensures data quality and consistency

2. **Research Scientist**
   - Analyzes the unified dataset
   - Provides insights on data distribution
   - Recommends optimal training parameters

3. **Model Trainer**
   - Fine-tunes the xLAM-v0.1 model
   - Implements training loop with recommended parameters
   - Monitors training progress

4. **Evaluator**
   - Tests model performance on various benchmarks
   - Provides detailed metrics and analysis
   - Ensures quality standards are met

5. **Deployment Specialist**
   - Packages the trained model for deployment
   - Sets up serving infrastructure
   - Manages deployment endpoints

6. **User Interaction Facilitator**
   - Provides user-friendly interface
   - Handles user queries and requests
   - Formats responses for clarity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the Salesforce AI Research team for the xLAM model
- Built with Streamlit for the interactive demo interface
- Inspired by the AgentOhana framework for unified agent training

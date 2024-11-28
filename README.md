# Legal Assistant with Generative AI

This project implements a legal research assistant using OpenAI's GPT models for analyzing case law and providing relevant legal insights. Done as part of assignment for DBA GenAI course of Golden Gate University.

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a .env file with your OpenAI API key: `OPENAI_API_KEY=your_key_here`
4. Place your legal cases dataset in `data/legal_cases.csv`

## Usage
Run the assistant:
```python
python src/legal_assistant.py

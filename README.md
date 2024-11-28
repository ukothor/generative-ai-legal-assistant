# Legal Assistant with Generative AI

This project implements a legal research assistant using OpenAI's GPT models for analyzing case law and providing relevant legal insights. Done as part of assignment for DBA GenAI course of Golden Gate University. Prior to implementation, I downloaded sample case data from the Harvard Law School Library's Caselaw Access Project https://case.law/ , which provides access to over 6.5 million U.S. court decisions. The initial data was in JSON format, which I converted to CSV using Python to make it more manageable for our analysis system.

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a .env file with your OpenAI API key: `OPENAI_API_KEY=your_key_here`
4. Place your legal cases dataset in `data/legal_cases.csv`

## Usage
Run the assistant:
```python
python src/legal_assistant.py

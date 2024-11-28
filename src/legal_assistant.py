import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

class LegalCaseAnalyzer:
    """
    A class to analyze legal cases using OpenAI's GPT models and embedding-based retrieval.
    
    This analyzer uses a combination of text embeddings for similarity search and GPT-4
    for generating insights about legal cases. It processes case law documents to find
    relevant precedents and generate analysis based on user queries.
    """
    
    def __init__(self) -> None:
        """Initialize the Legal Case Analyzer with OpenAI API credentials and logging."""
        # Load environment variables
        load_dotenv()
        
        # Set up OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        
        # Initialize data storage
        self.df = None
        self.embeddings: List = []
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self, filepath: str) -> None:
        """
        Load legal case data from a CSV file.
        
        Args:
            filepath: Path to the CSV file containing legal case data
        """
        try:
            self.df = pd.read_csv(filepath)
            self.logger.info(f"Loaded {len(self.df)} cases from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            raise

    def generate_embeddings(self, batch_size: int = 100) -> None:
        """
        Generate embeddings for all cases in the dataset.
        
        Args:
            batch_size: Number of cases to process in each batch
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        try:
            for i in range(0, len(self.df), batch_size):
                batch = self.df.iloc[i:i + batch_size]
                for text in batch['casebody']:
                    response = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text[:8000]  # Truncate to stay within token limit
                    )
                    self.embeddings.append(response.data[0].embedding)
                self.logger.info(f"Generated embeddings for batch {i//batch_size + 1}")
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def find_relevant_cases(self, query: str, top_k: int = 3) -> pd.DataFrame:
        """
        Find the most relevant cases for a given query using embedding similarity.
        
        Args:
            query: The search query
            top_k: Number of most relevant cases to return
            
        Returns:
            DataFrame containing the top_k most relevant cases
        """
        query_embedding = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        # Calculate similarities between query and all cases
        similarities = [
            np.dot(emb, query_embedding) / 
            (np.linalg.norm(emb) * np.linalg.norm(query_embedding))
            for emb in self.embeddings
        ]
        
        # Get indices of top_k most similar cases
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return self.df.iloc[top_indices]

    def generate_response(self, query: str, relevant_cases: pd.DataFrame) -> str:
        """
        Generate a response to the query based on relevant cases.
        
        Args:
            query: The user's query
            relevant_cases: DataFrame containing relevant cases to consider
            
        Returns:
            Generated response analyzing the cases in context of the query
        """
        # Format cases for context
        cases_context = "\n\n".join([
            f"Case: {row['name']}\n"
            f"Date: {row['decision_date']}\n"
            f"Court: {row['court']}\n"
            f"Summary: {row['casebody'][:1000]}"
            for _, row in relevant_cases.iterrows()
        ])

        prompt = f"""Based on these relevant cases:

{cases_context}

Question: {query}

Please provide a concise analysis referring to specific cases when relevant."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    def analyze_query(self, query: str) -> str:
        """
        Main method to analyze a legal query.
        
        Args:
            query: The user's legal query
            
        Returns:
            Analysis based on relevant cases
        """
        relevant_cases = self.find_relevant_cases(query)
        return self.generate_response(query, relevant_cases)

def main():
    """Main function to demonstrate the legal case analyzer."""
    try:
        # Get the directory containing the script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct path to data file
        data_path = os.path.join(os.path.dirname(current_dir), "data", "legal_cases.csv")
        
        # Initialize and run analyzer
        analyzer = LegalCaseAnalyzer()
        analyzer.load_data(data_path)
        analyzer.generate_embeddings()
        
        # Example query
        query = "What are the key arguments in property dispute cases?"
        response = analyzer.analyze_query(query)
        print(f"\nAnalysis:\n{response}")
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
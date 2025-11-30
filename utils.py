import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config

class MetricsCalculator:
    """
    Utility class: Handles Sentence-BERT embeddings and reward calculation metrics.
    Note: In a real scenario, this would load a real Toxicity Classifier and LLM for PPL calculation.
    For demonstration purposes, we use mock data or lightweight models.
    """
    def __init__(self):
        print(f"Loading Sentence-BERT model: {Config.SBERT_MODEL}...")
        self.sbert = SentenceTransformer(Config.SBERT_MODEL, device=Config.DEVICE)
        # Freeze parameters to speed up
        for param in self.sbert.parameters():
            param.requires_grad = False
            
    def get_embedding(self, text):
        """Get Sentence-BERT embedding"""
        with torch.no_grad():
            embedding = self.sbert.encode(text, convert_to_tensor=True)
        return embedding.cpu().numpy()

    def mock_toxicity_classifier(self, response_text, query_text):
        """
        Mock Toxicity Classifier (S_tox in the paper).
        In a real project, replace this with Roberta-toxicity or OpenAI Moderation API.
        Here we simply assume: if query contains 'bomb' and response doesn't say 'I cannot', it's toxic.
        """
        is_query_bad = "bomb" in query_text.lower() or "kill" in query_text.lower()
        is_refusal = "cannot" in response_text.lower() or "sorry" in response_text.lower()
        
        if is_query_bad and not is_refusal:
            return 0.9 # High toxicity
        elif is_query_bad and is_refusal:
            return 0.05 # Safe refusal
        else:
            return 0.0 # Benign
            
    def mock_perplexity(self, response_text):
        """
        Mock Perplexity (PPL).
        Real scenario requires calling LLM to compute loss.
        """
        # Simple mock: Random fluctuation
        return np.random.uniform(5.0, 20.0)

    def calculate_reward(self, query, response, toxicity_score, is_refusal, ppl):
        """
        Reward function from the paper:
        r_t = 0.4/(S_tox + 1) + 0.3*I_refuse - 0.2*PPL_norm + 0.1*Rel
        Note: PPL needs normalization as per paper. Simplified here.
        """
        # Relevance Rel (Simplified to 1.0, assuming model answer is always relevant)
        relevance = 1.0 
        
        # PPL Normalization (Assuming PPL range 0-100, map to 0-1)
        ppl_norm = min(ppl / 100.0, 1.0)
        
        # Refusal Indicator I_refuse (0 or 1)
        i_refuse = 1.0 if is_refusal else 0.0
        
        reward = (0.4 / (toxicity_score + 1)) + (0.3 * i_refuse) - (0.2 * ppl_norm) + (0.1 * relevance)
        return reward

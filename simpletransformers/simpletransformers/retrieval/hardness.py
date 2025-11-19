import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


class HardnessFeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.scaler = MinMaxScaler()
        self.fitted = False

    def fit(self, queries):
        self.vectorizer.fit(queries)
        tfidf_scores = np.asarray(self.vectorizer.transform(queries).sum(axis=1))
        self.scaler.fit(tfidf_scores)
        self.fitted = True

    def __call__(self, queries):
        if not self.fitted:
            # In a real scenario, you'd fit this on a large corpus or load a pre-fitted one.
            # For a small subset, we'll fit on the provided queries.
            self.fit(queries)
            # raise RuntimeError("HardnessFeatureExtractor must be fitted before use.")

        tfidf_scores = np.asarray(self.vectorizer.transform(queries).sum(axis=1))
        # Simple heuristic: higher TF-IDF sum implies less common words, potentially harder query
        # We can also incorporate query length, number of OOV words, etc.
        hardness_scores = 1 - self.scaler.transform(tfidf_scores).flatten()
        return hardness_scores


class HardnessPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid(),  # Output a score between 0 and 1
        )

    def forward(self, query_embedding):
        return self.predictor(query_embedding).squeeze(-1)


def normalize_hardness_array(hardness_array):
    min_val = np.min(hardness_array)
    max_val = np.max(hardness_array)
    if max_val == min_val:
        return np.zeros_like(hardness_array)
    normalized_array = (hardness_array - min_val) / (max_val - min_val)
    return normalized_array


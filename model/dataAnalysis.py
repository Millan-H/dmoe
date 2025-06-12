import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import cupy as cp
import time


class DataAnalysis:
    def __init__(self,referenceStructure=None):
        self.referenceStructure=referenceStructure
        self.encodedReferences=[]
        self.analyzer=SentenceTransformer("BAAI/bge-large-en-v1.5").to("cuda").half()
    def getEncoding(self, data):
        if self.referenceStructure!=None or self.encodedReferences!=None:
            encoding=[]
            if len(self.encodedReferences)==0:
                for domain, references in self.referenceStructure.items():
                    self.encodedReferences.append(torch.mean(torch.tensor(self.analyzer.encode(references)),dim=0))
            start=time.time()
            inputEncoded=self.analyzer.encode(data)
            similarities=f.cosine_similarity(
                torch.tensor(inputEncoded).unsqueeze(1), 
                torch.stack(self.encodedReferences).unsqueeze(0),
                dim=2
            )
            end=time.time()
            print(similarities)
            print(inputEncoded.shape)
            print(f"Time: {end-start}")
            return encoding
    def getStructure(self):
        return self.referenceStructure
    def getEncodedReferences(self):
        return self.encodedReferences
    def getBaseModel(self):
        return self.analyzer
    def trainEncoder(self,data,minConfidnece):
        for encoding,data in data:
            self.referenceStructure.append()
            self.encodedReferences.append(torch.mean(torch.tensor(self.analyzer.encode(data)),dim=0))

test = {
    "programming": [
        "def bubble_sort(arr): for i in range(len(arr)): for j in range(0, len(arr)-i-1):",
        "class DatabaseConnection: def __init__(self, host, port): self.connection = None",
        "try: result = api_call() except ConnectionError as e: logger.error(f'Failed: {e}')",
        "async function fetchData() { const response = await fetch('/api/users'); return response.json(); }",
        "if __name__ == '__main__': parser = argparse.ArgumentParser(); args = parser.parse_args()"
    ],
    
    "python": [
        "import pandas as pd; df = pd.read_csv('data.csv'); df.groupby('category').mean()",
        "with open('file.txt', 'r') as f: lines = [line.strip() for line in f.readlines()]",
        "from sklearn.linear_model import LinearRegression; model = LinearRegression().fit(X, y)",
        "lambda x: x**2 if x > 0 else -x**2; list(map(lambda x: x*2, [1,2,3,4,5]))",
        "import numpy as np; arr = np.random.randn(100, 50); eigenvals, eigenvecs = np.linalg.eig(arr)"
    ],
    
    "literature": [
        "The autumn leaves danced in the crisp morning air, whispering secrets of seasons past.",
        "She gazed upon the manuscript, its yellowed pages holding centuries of forgotten wisdom.",
        "His voice trembled with emotion as he recounted the tale of love lost and dreams shattered.",
        "The protagonist's journey through the labyrinthine city mirrored her own internal struggles.",
        "In the shadowy alcoves of the ancient library, time seemed to stand perfectly still."
    ],
    
    "conversational": [
        "Hey! How's your day going? I hope you're having a great time with your family.",
        "That sounds really interesting! Could you tell me more about what you're working on?",
        "I totally understand what you mean. I've been in a similar situation before myself.",
        "Thanks so much for your help! I really appreciate you taking the time to explain this.",
        "What do you think about trying that new restaurant downtown? I heard great reviews!"
    ],
    
    "machine_learning": [
        "The convolutional neural network achieved 94% accuracy on the ImageNet validation dataset.",
        "Gradient descent optimization with momentum converged faster than standard SGD approaches.",
        "Feature engineering and dimensionality reduction improved model performance significantly on high-dimensional data.",
        "Cross-validation revealed overfitting in the random forest model with default hyperparameters.",
        "The transformer architecture's self-attention mechanism enables parallel processing of sequential data."
    ]
}

data=["""What is the best Shakesperian play?""","""What is the best Shakesperian play?"""]
dataAnalysis=DataAnalysis(test)
dataAnalysis.getEncoding(data)
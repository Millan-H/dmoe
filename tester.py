import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import cupy as cp
import time


class DataAnalysis:
    def __init__(self,referenceStructure):
        self.referenceStructure=referenceStructure
        self.encodedReferences=[]
        self.analyzer=SentenceTransformer("BAAI/bge-large-en-v1.5").to("cuda").half()
    def getEncoding(self, data):
        start=time.time()
        inputEncoded=self.analyzer.encode(data, convert_to_tensor=True, device="cuda").half()
        end=time.time()
        print(inputEncoded.shape)
        print(f"Time: {end-start}")
    def getStructure(self):
        return self.encodingStructure

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

data=[
# Programming (10 examples)
"for (int i = 0; i < array.length; i++) { sum += array[i]; }",
"function calculateDistance(x1, y1, x2, y2) { return Math.sqrt((x2-x1)**2 + (y2-y1)2); }",
"while (queue.isEmpty() == false) { node = queue.dequeue(); processNode(node); }",
"if (user.isAuthenticated()) { redirectTo('/dashboard'); } else { showLoginForm(); }",
"const hashMap = new Map(); hashMap.set(key, value); return hashMap.get(key);",
"recursiveFunction(n) { if (n <= 1) return 1; else return n * recursiveFunction(n-1); }",
"try { connection.execute(query); } catch (SQLException e) { rollback(); }",
"struct Node { int data; Node next; Node prev; };",
"SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id",
"algorithm quickSort(arr, low, high): if low < high then partition and recurse",
# Python (10 examples)  
"import matplotlib.pyplot as plt; plt.plot(x_data, y_data); plt.show()",
"df['new_column'] = df.apply(lambda row: row['col1'] + row['col2'], axis=1)",
"with sqlite3.connect('database.db') as conn: cursor = conn.execute(query)",
"class Person: def __init__(self, name, age): self.name = name; self.age = age",
"[x**2 for x in range(10) if x % 2 == 0]",
"import requests; response = requests.get('https://api.example.com/data')",
"from collections import defaultdict; word_count = defaultdict(int)",
"np.where(array > threshold, array, 0)",
"pickle.dump(model, open('trained_model.pkl', 'wb'))",
"async def fetch_data(): async with aiohttp.ClientSession() as session:",

# Literature (10 examples)
"The moonlight cascaded through the ancient oak's gnarled branches, casting ethereal shadows.",
"Her heart ached with the weight of unspoken words and memories long buried.",
"In the distance, the cathedral bells echoed across the cobblestone streets of the old city.",
"Time seemed to slow as she turned the yellowed pages of her grandmother's diary.",
"The storm raged outside while inside, by the fireplace, stories came alive through whispered words.",
"His eyes held the depth of oceans and the mystery of forgotten civilizations.",
"The garden bloomed with roses that carried secrets of love letters never sent.",
"Through the mist emerged a figure cloaked in velvet, walking toward an uncertain destiny.",
"The library's silence was broken only by the gentle rustling of ancient manuscripts.",
"She painted her sorrows in watercolors that bled like tears across the canvas.",

# Conversational (10 examples)
"Hey there! How was your weekend? Did you get up to anything fun?",
"That's awesome! I'd love to hear more about your trip to Japan sometime.",
"No worries at all! Thanks for letting me know you're running a bit late.",
"What do you think about grabbing coffee this afternoon if you're free?",
"I totally get what you mean - I've been in the exact same situation before.",
"Congrats on the promotion! You really deserve it after all your hard work.",
"Hope you're feeling better today! Let me know if you need anything at all.",
"That sounds like such a great idea! Count me in if you need an extra person.",
"Thanks so much for helping me out with this - I really appreciate your time.",
"How's your family doing? I hope everyone is staying healthy and happy.",

# Machine Learning (10 examples)
"The neural network achieved 97.2% accuracy on the test dataset after hyperparameter tuning.",
"Support vector machines with RBF kernels performed better than linear classifiers on this problem.",
"Cross-validation revealed significant overfitting when using too many hidden layers.",
"Feature selection using mutual information reduced dimensionality from 10,000 to 500 features.",
"The random forest model showed lower variance compared to individual decision trees.",
"Batch normalization improved convergence speed during backpropagation training.",
"Transfer learning from pretrained ImageNet models accelerated training on our custom dataset.",
"The attention mechanism in transformers enables parallel processing of sequence data.",
"Regularization techniques like dropout and L2 penalty reduced model overfitting significantly.",
"Ensemble methods combining multiple weak learners outperformed single strong classifiers."]
dataAnalysis=DataAnalysis(test)
dataAnalysis.getEncoding(data)
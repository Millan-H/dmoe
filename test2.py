from bertopic import BERTopic
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import time


data=load_dataset("imdb")["train"]["text"]
model=SentenceTransformer("all-MiniLM-L6-v2",device="cuda")

topicClassifier=BERTopic(embedding_model=model,calculate_probabilities=True)
start=time.time()
sorted,probabilities=topicClassifier.fit_transform(data)
end=time.time()
print(sorted)
print(probabilities)
info=topicClassifier.get_topic_info()
labels=topicClassifier.generate_topic_labels()
embeddings=topicClassifier.topic_embeddings_
print(info)
print(labels)
print(embeddings)
print(f"Time: {end-start}")
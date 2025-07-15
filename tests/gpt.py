import sys
import os
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, AutoModel
from torch.utils.data import IterableDataset
from datasets import load_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model.preProcesser import DataHandler
from model.controller import DMoE

tokenizer=GPT2Tokenizer(GPT2Config()).to("cuda").half()
gpt=GPT2LMHeadModel(GPT2Config()).to("cuda").half()
bert=AutoModel.from_pretrained("bert-base-cased").to("cuda").half()
data=load_dataset("bigcode/the-stack-v2",streaming=True)
print(data)

dataExperts=[]

dmoe=DMoE(bert,gpt,512,[1 for i in range(512)],0.8,16,0.3)
dmoe.trainingPipeline(data.take(512000))

current = []
tokens = 0
chunkNumber = 0

for datum in data['train']:
    content = datum['content']
    tokensTotal = len(tokenizer.encode(content))
    
    if tokens + tokensTotal > 512000:
        if current:
            dmoe.trainingPipeline(current)
        
        current = [content]
        tokens = tokens
    else:
        current.append(content)
        tokens += tokens

if current:
    dmoe.trainingPipeline(current)

#train base on linguistics -> organize dataset in order of decreasing similarity from area with the least data -> retrain on cached data that doesn't fit a domain -> more epochs from step 3 if needed or go from step 2 and do a new dataset
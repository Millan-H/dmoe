import sys
import os
from controller import DMoE
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, AutoModel

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model.preProcesser import DataHandler


gpt=GPT2LMHeadModel(GPT2Config())
bert=AutoModel.from_pretrained("bert-base-cased")
testData=[]

dataAnalyzers=[]
dataExperts=[]
dmoe=DMoE(bert,gpt,512,[1 for i in range(512)],0.8,16,0.3)
dmoe.trainingPipeline(testData)

#train base on linguistics -> organize dataset in order of decreasing similarity from area with the least data -> retrain on cached data that doesn't fit a domain -> more epochs from step 3 if needed or go from step 2 and do a new dataset
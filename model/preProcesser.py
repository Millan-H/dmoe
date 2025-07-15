from model.dataAnalysis import DataAnalysis
from model.classifier import Classifier
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import torch
from torch.optim import Adam
import cupy as cp

class DataHandler:
    def __init__(self, dataAnalysis: DataAnalysis, classifier: Classifier,threshold):
        self.dataAnalysis=dataAnalysis
        self.classifier=classifier
        self.reformattedData=[]
        self.baseModel=dataAnalysis.getBaseModel()
        self.topicClassifier=BERTopic(embedding_model=self.baseModel)
    def getDataEncodings(self,data):
        returnData=[[None,[datum]] for datum in data]
        for i in range(len(data)):
            returnData[i][0]=self.dataAnalysis.getEncoding(data[i])
    def sortData(self,data):
        topics,probablities=self.topicClassifier.fit_transform(data)
        topicsDict=defaultdict(list)
        lowConfidence=defaultdict(list)
        for topic,probability,document in zip(topics,probablities,data):
            if probability>0.7:
                topicsDict[topic].append(document)
            else:
                lowConfidence[topic].append(document)
        self.dataAnalysis.trainEncoder(topicsDict)
        self.dataAnalysis.getEncoding(data)
        return topicsDict
    def trainBaseModel(self, data, baseSize, batchSize):
        indecies=torch.random.sample(range(len(data)),baseSize)
        randomData=torch.tensor(data, dtype=torch.bfloat16)[indecies]
        baseModel=GPT2LMHeadModel(GPT2Config()).to("cuda").half()
        tokenizer=GPT2Tokenizer.from_pretrained("gpt")
        optimizer=Adam(baseModel.parameters(),lr=0.001)
        for i in range(1, len(randomData), batchSize):
            batchedData=randomData[i:i+batchSize]
            texts=[x[1] for x in batchedData]
            tokenized = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            for k in tokenized:
                tokenized[k] = tokenized[k].to("cuda")
            tokenized["input_ids"] = tokenized["input_ids"].half()
            labels=tokenized["input_ids"].clone().to(torch.long)
            labels[tokenized["attention_mask"] == 0]=-100
            output=baseModel(**tokenized, labels=labels)
            loss=output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.classifier.setBaseModel("base",baseModel.to(dtype=torch.bfloat16).state_dict())

preProcesser=DataHandler()
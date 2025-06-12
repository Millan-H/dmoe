from model.dataAnalysis import DataAnalysis
from model.classifier import Classifier
from model.memory import MemoryHandler
from model.preProcesser import DataHandler
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

class DMoE:
    def __init__(self, analyzerStructure, expertStructure, analyzerCount, weights, threshold, expertMergeCount, cutThreshold, encodingStucture=None):
        self.dataAnalysis=DataAnalysis(analyzerStructure, analyzerCount,encodingStucture)
        self.classifier=Classifier(expertStructure, weights,threshold,expertMergeCount,cutThreshold)
        self.memory=MemoryHandler()
        self.losses={}
        self.optimizers={}
        self.tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
        self.expert=GPT2LMHeadModel(GPT2Config())
        self.expert.half().to("cuda")
        self.cache={}
        self.dataHandler=DataHandler(self.dataAnalysis,self.classifier)
    def run(self, data, predefinedEncoding=None):
        if predefinedEncoding==None:
            encodingUnclassified=self.dataAnalysis.getEncoding(data)
            encoding=self.classifier.getExpert(encodingUnclassified)
            if encoding not in self.cache.keys():
                self.expert.load_state_dict(self.memory.getExpertFromEncoding(encoding)).to("cuda").half()
                self.cache[encoding]=self.expert
        else:
            encoding=predefinedEncoding
        self.expert=self.cache[encoding]
        tokenizedData=self.tokenizer(data,return_tensors="pt").to("cuda").half()
        output=self.expert(tokenizedData)
        return output
    def train(self,data,batchSize=128,full=True,baseModelDataSize=1000000000):
        if full:
            data=self.trainingPipeline(data,batchSize,baseModelDataSize)
        expertEncoding=self.classifier.getExpert(data[0])
        if expertEncoding not in self.cache.keys():
            self.expert.load_state_dict(self.memory.getExpertFromEncoding(expertEncoding)).to("cuda").half()
            self.cache[expertEncoding]=self.expert
        if expertEncoding not in self.losses:
            self.losses[expertEncoding]=CrossEntropyLoss()
        loss=self.losses[expertEncoding]
        if expertEncoding not in self.optimizers:
            self.optimizers[expertEncoding]=Adam(expert.parameters(), lr=0.001)
        optimizer=self.optimizers[expertEncoding]
        for i in range(0,len(data[1]),batchSize):
            batch=data[1][i:i+batchSize]
            expert=self.cache[expertEncoding]
            texts=[x[1] for x in batch]
            tokenized = self.tokenizer(
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
            output=expert(**tokenized, labels=labels)
            loss=output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    def trainingPipeline(self, data, batchSize, baseModelBaseDataSize):
        self.dataHandler.trainBaseModel(data, baseModelBaseDataSize, batchSize)
        sortedData=self.dataHandler.sortData(data)
        return sortedData
    def getDataAnalysis(self):
        return self.dataAnalysis
    def getClassifier(self):
        return self.classifier
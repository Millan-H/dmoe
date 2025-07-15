from dataAnalysis import DataAnalysis
from classifier import Classifier
from memory import MemoryHandler
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import time

class DMoE:
    def __init__(self, threshold,cutThreshold, expertMergeCount,  encodingStucture=None):
        self.dataAnalysis=DataAnalysis()
        self.classifier=Classifier(threshold,cutThreshold,expertMergeCount)
        self.memory=MemoryHandler()
        self.losses={}
        self.optimizers={}
        self.tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
        self.expert=GPT2LMHeadModel(GPT2Config())
        self.expert.half().to("cuda")
        self.cache={}
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
        output=self.expert.generate(tokenizedData,maxLength=512,pad_token_id=self.tokenizer.eos_token_id)
        return output
    def train(self,data,batchSize=512,full=True,baseModelDataSize=1000000000):
        if full:
            data=self.trainingPipeline(data,batchSize,baseModelDataSize)
        for encoding in data.keys():
            if encoding!='cache':
                if len(self.memory.getExpertEncodings()):
                    self.memory.writeExpert(encoding,)
                encoding=self.classifier.getExpert(encoding)
                self.expert.to("cuda").half().load_state_dict(self.memory.getExpertFromEncoding(encoding))
                loss=CrossEntropyLoss()
                optimizer=Adam(self.expert.parameters(), lr=0.001)
                for i in range(0,len(data[encoding]),batchSize):
                    start=time.time()
                    batch=data[1][i:i+batchSize]
                    texts=[x[1] for x in batch]
                    tokenized=self.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    tokenized["input_ids"] = tokenized["input_ids"].to("cuda")
                    attentionMask = tokenized["attention_mask"].to("cuda")
                    labels = tokenized["input_ids"].clone()
                    labels[attentionMask == 0] = -100
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        outputs = self.expert(input_ids=tokenized["input_ids"], attention_mask=attentionMask, labels=labels)
                        loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    totalLoss += loss.item()
                    numBatches += 1
                    end=time.time()
                    if i % (batchSize * 10) == 0:
                        print(f"Batch {i//batchSize}, Loss: {loss.item():.4f}, Time: {end-start}s")
            else:
                avgLoss = totalLoss / numBatches if numBatches > 0 else 0
                print(f"Training completed. Average loss: {avgLoss}")
                break
        
    def trainingPipeline(self, data, batchSize, baseModelBaseDataSize):
        self.dataHandler.trainBaseModel(data, baseModelBaseDataSize, batchSize)
        sortedData=self.dataHandler.sortData(data)
        self.train(data)
        return sortedData
    def getDataAnalysis(self):
        return self.dataAnalysis
    def getClassifier(self):
        return self.classifier

dmoe=DMoE(0.5,0.5,0.5,0.5)
preprocesser=DataAnalysis()
data=preprocesser.preprocessData(cluster=False,restriction=10000)
dmoe.train(data,full=False)
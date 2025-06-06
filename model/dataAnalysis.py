import torch
from torch.nn import MSELoss
from torch.optim import Adam
import threading
import time

class DataAnalysis:
    def __init__(self,analyzerStructure,analyzerCount,encodingStucture=None):
        self.analyzers=[analyzerStructure for i in range(analyzerCount)]
        self.encodingStructure=encodingStucture
        self.grads=[[] for analyzer in analyzerCount]
    def getEncodings(self, data):
        encoding=[]
        for analyzer in self.analyzers:
            encoding.append(analyzer.forward(torch.tensor(data)))
        return encoding
    def getStructure(self):
        return self.encodingStructure
    def train(self,data,batchSize=128):
        threads=[]
        losses=[MSELoss() for analyzer in self.analyzers]
        for i in range(len(data),batchSize):
            for j in range(len(data[i:i+batchSize])):
                for k in range(len(self.analyzers)):
                    threads.append(threading.Thread(None,self.trainIndividual(data[j], losses[k], k)))
                map(threading.Thread.start(), threads)
                #close threads
            for k in range(len(self.analyzers)):
                grads=sum(self.grads[k])/batchSize
                threads.append(threading.Thread(None,self.backprop(grads, k)))
            map(threading.Thread.start(), threads)
            #close threads
    def getGradsIndividual(self,datum,loss,index): #check this
        analyzer=self.analyzers[index]
        optimizer=Adam(analyzer.parameters(),lr=0.001)
        loss(analyzer.forward(datum[1]),torch.tensor(datum[0][index]))
        optimizer.zero_grad()
        grads=loss.backward()
        self.grads[index].append(grads)
    def backprop(self,grad,index): #check this
        analyzer=self.analyzers[index]
        analyzer.backward(grad)
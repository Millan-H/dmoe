from model.dataAnalysis import DataAnalysis
from model.classifier import Classifier
from model.memory import MemoryHandler

class DMoE:
    def __init__(self, analyzerStructure, expertStructure, analyzerCount, weights, threshold, expertMergeCount, cutThreshold, encodingStucture=None):
        self.dataAnalysis=DataAnalysis(analyzerStructure, analyzerCount,encodingStucture)
        self.classifier=Classifier(expertStructure, weights,threshold,expertMergeCount,cutThreshold)
        self.memory=MemoryHandler()
    def run(self, data):
        encodings=self.dataAnalysis.getEncodings(data)
        expert=self.memory.read(self.classifier.getExpert(encodings[0]))
        if encodings[1]!=None:
            expert=None #implement initializaiton with parameters (may need to use custom implementation for this)
        output=expert.forward(data)
        return output
    def trainExperts(self,data,batchSize=128):
        experts=self.memory.getExpertEncodings()
        for i in range(len(data),batchSize):
            for j in range(i,i+batchSize):
                encodings=self.dataAnalysis.getEncodings(data)
                expert=self.memory.read(self.classifier.getExpert(encodings[0]))
                if encodings[1]!=None:
                    expert=None #implement initializaiton with parameters (may need to use custom implementation for this)
                output=expert.forward(data)
            
    def trainAnalysis(self,data,batchSize=128):
        self.dataAnalysis.train(data,batchSize)
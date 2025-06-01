from classifier import Classifier
from ann import TransformerNN, Network
from dataAnalysis import DataAnalysis

class DMoE:
    def __init__(self, encoder, networkType, networkStructure):
        self.dataAnalysis=DataAnalysis()
        self.classifier=Classifier(encoder, networkType)
        self.networks=[]
        self.networks.append(self.classifer.getNetworks())
        self.networkType=networkType
    def run(self, rawData):
        pass
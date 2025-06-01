class Classifier:
    def __init__(self, encoderStructure, networkType): #encoderStruture as a list of labels
        self.encoder=encoderStructure
        self.networks={}
        self.networkType=networkType
        self.weights=[0 for i in range(len(self.encoder))]
    def classifyNetwork(self, rawData, processedData): #processData as a dict or list relating processed data to encoder labels
        pass
    def addNetwork(self, encoding, network):
        self.networks[encoding]=network.getWeights()
    def getNetworks(self):
        return self.networks
    def getEncoder(self):
        return self.encoder
    def train(self, data): 
        pass

    
import json
import torch
import ast
from numpy._core.multiarray import _reconstruct

class MemoryHandler:
    def __init__(self):
        pass
    def writeExpert(self,encoding,parameters):
        memoryData=torch.load('C:/Users/milla/dmoe/model/memory.pt')
        memoryData[f"{encoding.tolist()}"]=parameters
        torch.save(memoryData,'./model/memory.pt')
        return "Success"
    def getExpertFromEncoding(self, encoding):
        return torch.load('./model/memory.pt')[f"{encoding}"]
    def getExpertEncodings(self):
        data=torch.load('./model/memory.pt')
        return list(map(ast.literal_eval,list(data.keys())))
    def getEncodingStructure(self):
        with open('./model/encodingStructure.json','r') as file:
            data=json.load(file)["encodingStructure"]
            for i,datum in enumerate(data):
                data[i]=torch.tensor(datum,dtype=torch.bfloat16).to("cuda")
        return data
    def updateEncodingStructure(self,newStructure):
        torch.save(newStructure,'./model/encodingStructure.json')
        return "Success"
    def updateSampleCheckpoint(self,sample):
        try:
            torch.save(sample,'C:/Users/milla/dmoe/model/sample.pt')
            return "Success"
        except Exception as e:
            print(f"Sample Checkpoint Write Error: {e}")

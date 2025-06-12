import json
from transformers import GPT2LMHeadModel, GPT2Config
import torch
import ast

class MemoryHandler:
    def __init__(self):
        pass
    def writeExpert(self,encoding,parameters):
        memoryData=torch.load('C:/Users/milla/dmoe/model/memory.pt')
        memoryData[f"{encoding}"]=parameters
        torch.save(memoryData,'./model/memory.pt')
        return "Success"
    def getExpertFromEncoding(self, encoding):
        return torch.load('./model/memory.pt')[f"{encoding}"]
    def getExpertEncodings(self):
        data=torch.load('./model/memory.pt')
        return list(map(ast.literal_eval,list(data.keys())))

print(MemoryHandler().writeExpert("asdf",torch.tensor([1],dtype=torch.bfloat16)))
print(MemoryHandler().getExpertFromEncoding("asdf"))
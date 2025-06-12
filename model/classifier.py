from memory import MemoryHandler
from transformers import GPT2LMHeadModel, GPT2Config
import torch
import time

class Classifier:
    def __init__(self, weights, threshold, dmodel, expertMergeCount, cutThreshold):
        if weights!=None:
            self.weights=torch.tensor(weights,dtype=torch.bfloat16).to("cuda")
        else:
            self.weights=torch.tensor([1 for i in range(dmodel)],dtype=torch.bfloat16).to("cuda")
        self.threshold=threshold
        self.expertMergeCount=expertMergeCount
        self.cutThreshold=cutThreshold
        self.firstRun=False
        self.baseModel=None
    def getExpert(self, encoding, data=None, mode="prod"):
        memory=MemoryHandler()
        encoding=torch.tensor(encoding,dtype=torch.bfloat16).to("cuda")
        experts=memory.getExpertEncodings()
        self.experts=torch.tensor(experts,dtype=torch.bfloat16).to("cuda")
        if len(self.experts)==0:
            self.experts=torch.tensor([encoding],dtype=torch.bfloat16).to("cuda")
            startExpert=GPT2LMHeadModel(GPT2Config()).to("cuda")
            memory.writeExpert(encoding,startExpert.to(dtype=torch.bfloat16).state_dict())
        differences=self.experts-encoding.unsqueeze(0)
        comparisons=torch.mean(torch.abs(differences*self.weights), axis=1)
        bestExpertIdx=torch.argmin(comparisons)
        bestExpert=comparisons[bestExpertIdx]
        bestExpertEncoding=self.experts[bestExpertIdx]
        bestExpertDist=1-torch.mean(abs((bestExpertEncoding-encoding)/torch.maximum(bestExpertEncoding+1e-8, encoding+1e-8)))
        if bestExpertDist<self.threshold:
            if self.baseModel!=None:
                mergingExperts=[bestExpertEncoding]
                comparisonCopy=comparisons.clone()
                comparisonCopy=torch.cat([comparisonCopy[:int(bestExpertIdx)], comparisonCopy[int(bestExpertIdx)+1:]])
                if len(comparisonCopy)>0:
                    remainingScores = comparisons
                    sortedIndices = torch.argsort(remainingScores, descending=True)
                    numAdditionalExperts = min(len(sortedIndices), self.expertMergeCount - 1)
                    for idx in sortedIndices[:numAdditionalExperts]:
                        mergingExperts.append(experts[int(idx)])
                    mergingExperts=torch.tensor(mergingExperts,dtype=torch.bfloat16).to("cuda")
                mergingExpertsFiltered=self.filterExperts(mergingExperts,encoding)
                if len(mergingExpertsFiltered)==0:
                    if mode == "prod":
                        mergingExpertsFiltered=bestExpertEncoding
                    elif mode == "train":
                        self.cacheData(encoding, data)
                        return self.baseModel[0]
                newExpert=self.mergeExperts(mergingExpertsFiltered).to("cuda")
                memory.writeExpert(encoding,newExpert.to(dtype=torch.bfloat16).state_dict())
                return encoding
            else:
                return encoding
        else:
            return bestExpertEncoding
    def filterExperts(self,experts,encoding):
        if len(experts) <= 1:
            return experts
        self.experts = torch.tensor(experts,dtype=torch.bfloat16).to("cuda")
        expertDists=torch.mean(abs((experts-encoding)/torch.maximum(experts+1e-8, encoding+1e-8)))
        return self.experts[expertDists > self.cutThreshold]
    def cacheData(self,encoding,data):
        try:
            cachedData=torch.load("C:/Users/milla/dmoe/model/dataCache.pt")
        except:
            cachedData = {}
        cachedData[tuple(encoding.cpu().tolist())] = data
        torch.save(cachedData,"C:/Users/milla/dmoe/model/dataCache.pt")
    def mergeExperts(self, mergingExperts, noiseThreshold=1e-6, consensusThreshold=0.8):
        if len(mergingExperts)<=1:
            return mergingExperts
        else:
            mergedExpert = GPT2LMHeadModel(GPT2Config())
            memory=MemoryHandler()
            baseStateDict = self.baseModel[1].to("cuda").half().state_dict()
            mergedStateDict = {}
            expertStateDicts = [memory.getExpertFromEncoding(expert) for expert in mergingExperts]
            numExperts = len(expertStateDicts)
            for paramName, baseParam in baseStateDict.items():
                expertParams = torch.stack([
                    expertDict[paramName] for expertDict in expertStateDicts
                ])
                paramDiffs = expertParams - baseParam.unsqueeze(0)
                significantMask = torch.abs(paramDiffs) > noiseThreshold
                paramSigns = torch.sign(paramDiffs)
                paramSigns = torch.where(significantMask, paramSigns, torch.zeros_like(paramSigns))
                positiveVotes = (paramSigns > 0).sum(dim=0).float()
                negativeVotes = (paramSigns < 0).sum(dim=0).float()
                totalVotes = positiveVotes + negativeVotes
                consensusPositive = positiveVotes >= negativeVotes
                hasConsensus = (torch.max(positiveVotes, negativeVotes) / numExperts) >= consensusThreshold
                strongConsensus = hasConsensus & (totalVotes > 0)
                hasConflict = ~hasConsensus & (totalVotes > 0)
                noChanges = totalVotes == 0
                mergedParam = baseParam.clone()
                if strongConsensus.any():
                    consensusMask = strongConsensus & significantMask
                    consensusDiffs = torch.where(consensusMask, paramDiffs, torch.zeros_like(paramDiffs))
                    avgConsensus = consensusDiffs.sum(dim=0) / consensusMask.sum(dim=0).clamp(min=1)
                    mergedParam = torch.where(strongConsensus, baseParam + avgConsensus, mergedParam)
                if hasConflict.any():
                    weights = torch.abs(paramDiffs) * significantMask.float()
                    weightedDiffs = paramDiffs * weights
                    totalWeights = weights.sum(dim=0).clamp(min=1e-12)
                    weightedAvg = weightedDiffs.sum(dim=0) / totalWeights
                    mergedParam = torch.where(hasConflict, baseParam + weightedAvg, mergedParam)
                mergedStateDict[paramName] = mergedParam
            mergedExpert.load_state_dict(mergedStateDict)
            return mergedExpert
    def setBaseModel(self,encoding,model):
        self.baseModel=[encoding,model.to("cuda").half()]

classifier=Classifier(None,0.8,5,10,0.4)
classifier.setBaseModel([1,3,0,2,4],GPT2LMHeadModel(GPT2Config()))
start=time.time()
print(classifier.getExpert([1,2,1,1,2],"asdf"))
end=time.time()
print(f"Time: {end-start}")
from memory import MemoryHandler
from transformers import GPT2LMHeadModel, GPT2Config
import torch
import torch.nn.functional as f
import cupy as cp
import time

class Classifier:
    def __init__(self, threshold, cutThreshold, expertMergeCount):
        self.threshold=threshold
        self.expertMergeCount=expertMergeCount
        self.cutThreshold=cutThreshold
        self.firstRun=False
        self.baseModel=None
        memory=MemoryHandler()
        experts=memory.getExpertEncodings()
        self.experts=torch.tensor(experts,dtype=torch.bfloat16).to("cuda")
        self.currentTrainEncoding=None
        self.currentTrainExpert=None
        self.sortEncodings=None
    def getExpert(self, encoding, data=None, mode="prod"):
        memory=MemoryHandler()
        encoding=torch.tensor(encoding,dtype=torch.bfloat16).to("cuda")

        if len(self.experts)==0:
            self.experts=torch.tensor([encoding],dtype=torch.bfloat16).to("cuda")
            startExpert=GPT2LMHeadModel(GPT2Config()).to("cuda")
            memory.writeExpert(encoding,startExpert.to(dtype=torch.bfloat16).state_dict())
            experts=memory.getExpertEncodings()
            self.experts=torch.tensor(experts,dtype=torch.bfloat16).to("cuda")
        
        comparisons=torch.cdist(torch.tensor(encoding,dtype=torch.float32).unsqueeze(0), torch.tensor(self.experts,dtype=torch.float32), p=2)
        viableExperts=comparisons<self.threshold
        trueCount=viableExperts.tolist().count(True)
        if trueCount>=1:
            if trueCount>=self.expertMergeCount:
                sortedComparisons=torch.sort(comparisons[viableExperts[0]], descending=True)[0]
                viableExperts=comparisons==sortedComparisons[0:self.expertMergeCount]
            return self.mergeExperts(self.experts[viableExperts[0]])
        else:
            bestExpertIdx=torch.argmin(comparisons)
            bestExpertEncoding=self.experts[bestExpertIdx]
            if self.baseModel!=None:
                mergingExperts=[bestExpertEncoding]
                comparisonCopy=comparisons.clone()
                comparisonCopy=torch.cat([comparisonCopy[:int(bestExpertIdx)], comparisonCopy[int(bestExpertIdx)+1:]])
                if len(comparisonCopy)>0:
                    remainingScores = comparisons
                    sortedIndices = torch.argsort(remainingScores, descending=True)
                    numAdditionalExperts = min(len(sortedIndices), self.expertMergeCount - 1)
                    for idx in sortedIndices[:numAdditionalExperts]:
                        mergingExperts.append(torch.tensor(self.experts[int(idx)],dtype=torch.bfloat16).to("cuda"))
                    mergingExperts=torch.stack(mergingExperts).to("cuda")
                mergingExpertsFiltered=self.filterExperts(mergingExperts,encoding)
                if len(mergingExpertsFiltered)==0:
                    return self.baseModel[1]
                newExpert=self.mergeExperts(mergingExpertsFiltered).to("cuda")
                memory.writeExpert(encoding,newExpert)
                experts=memory.getExpertEncodings()
                self.experts=torch.tensor(experts,dtype=torch.bfloat16).to("cuda")
                return encoding
            else:
                return encoding
    def getTrainExpert(self, encoding):
        if encoding!=self.currentTrainEncoding:
            self.currentTrainEncoding=torch.tensor(encoding,dtype=torch.bfloat16).to("cuda")
            memory=MemoryHandler()
            comparisons=f.cosine_similarity(self.currentTrainEncoding, self.experts, dim=1)
            bestExpertIdx=torch.argmax(comparisons)
            bestExpertEncoding=self.experts[bestExpertIdx]
            mergingExperts=[bestExpertEncoding]
            comparisonCopy=comparisons.clone()
            comparisonCopy=torch.cat([comparisonCopy[:int(bestExpertIdx)], comparisonCopy[int(bestExpertIdx)+1:]])
            if len(comparisonCopy)>0:
                remainingScores = comparisons
                sortedIndices = torch.argsort(remainingScores, descending=True)
                numAdditionalExperts = min(len(sortedIndices), self.expertMergeCount - 1)
                for idx in sortedIndices[:numAdditionalExperts]:
                    mergingExperts.append(torch.tensor(self.experts[int(idx)],dtype=torch.bfloat16).to("cuda"))
                mergingExperts=torch.stack(mergingExperts).to("cuda")
            mergingExpertsFiltered=self.filterExperts(mergingExperts,encoding)
            newExpert=self.mergeExperts(mergingExpertsFiltered).to("cuda")
            memory.writeExpert(encoding,newExpert)
            experts=memory.getExpertEncodings()
            self.experts=torch.tensor(experts,dtype=torch.bfloat16).to("cuda")
            self.currentTrainExpert=newExpert
        return self.currentTrainExpert
    def sort(self, encoding):
        memory=MemoryHandler()
        experts=[]
        encoding=torch.tensor(encoding,dtype=torch.bfloat16).to("cuda")
        for sortEncoding in self.sortEncodings:
            cudaedSortEncoding=torch.tensor(sortEncoding).to("cuda").half()
            if not torch.equal(cudaedSortEncoding,encoding):
                experts.append(cudaedSortEncoding)
        experts=torch.stack(experts).to("cuda").half()
        comparisons=torch.cdist(torch.tensor(encoding,dtype=torch.float32).unsqueeze(0), torch.tensor(experts,dtype=torch.float32), p=2)[0]
        viableExperts=comparisons<self.threshold
        trueCount=viableExperts.tolist().count(True)
        if trueCount>=1:
            if trueCount>=self.expertMergeCount:
                sortedComparisons=torch.sort(comparisons[viableExperts[0]], descending=True)[0]
                viableExperts=comparisons==sortedComparisons[0:self.expertMergeCount]
            return self.sortEncodings[viableExperts[0]]
        else:
            bestExpertIdx=torch.argmin(comparisons)
            bestExpertEncoding=experts[bestExpertIdx]
            mergingExperts=[bestExpertEncoding]
            comparisonCopy=comparisons.clone()
            comparisonCopy=torch.cat([comparisonCopy[:int(bestExpertIdx)], comparisonCopy[int(bestExpertIdx)+1:]])
            if len(comparisonCopy)>0:
                remainingScores = comparisons
                sortedIndices = torch.argsort(remainingScores, descending=True)
                numAdditionalExperts = min(len(sortedIndices), self.expertMergeCount - 1)
                for idx in sortedIndices[:numAdditionalExperts]:
                    mergingExperts.append(torch.tensor(experts[int(idx)],dtype=torch.bfloat16).to("cuda"))
                mergingExperts=torch.stack(mergingExperts).to("cuda")
            mergingExpertsFiltered=self.filterExperts(mergingExperts,encoding)
            if len(mergingExpertsFiltered)==0:
                print('base')
                return []
            return mergingExpertsFiltered
    def getCacheExpert(self, encoding):
        memory=MemoryHandler()
        encoding=torch.tensor(encoding,dtype=torch.bfloat16).to("cuda")

        if len(self.experts)==0:
            self.experts=torch.tensor([encoding],dtype=torch.bfloat16).to("cuda")
            startExpert=GPT2LMHeadModel(GPT2Config()).to("cuda")
            memory.writeExpert(encoding,startExpert.to(dtype=torch.bfloat16).state_dict())
            experts=memory.getExpertEncodings()
            self.experts=torch.tensor(experts,dtype=torch.bfloat16).to("cuda")
        
        comparisons=torch.cdist(torch.tensor(encoding,dtype=torch.float32).unsqueeze(0), torch.tensor(self.experts,dtype=torch.float32), p=2)

        return self.experts[torch.argmin(comparisons)]
    def setSortEncodings(self, encodings):
        self.sortEncodings=encodings
    def filterExperts(self,experts,encoding,trainingMode=False):
        if len(experts) <= 1:
            return experts
        self.experts = torch.tensor(experts,dtype=torch.bfloat16).to("cuda")
        expertDists=torch.mean(abs((experts-encoding)/torch.maximum(experts+1e-8, encoding+1e-8)))
        if not trainingMode:
            return self.experts[expertDists < self.cutThreshold]
        else:
            return self.experts[expertDists < self.threshold]
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
            print('yayy')
            memory=MemoryHandler()
            baseStateDict = self.baseModel[1]
            mergedStateDict = {}
            expertStateDicts = mergingExperts
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
            return mergedStateDict
    def setBaseModel(self,encoding,model):
        self.baseModel=[encoding,model.to("cuda").half().state_dict()]


'''classifier=Classifier(4.8,5,5)
encoding=torch.rand((1,150))[0]
print(encoding)
classifier.getExpert(encoding)'''
'''classifier=Classifier(None,0.85,1000,10,0.5)
classifier.setBaseModel(torch.randint(0,10,(1000, )),GPT2LMHeadModel(GPT2Config()))
print("asdfasdf")
start=time.time()
classifier.mergeExperts([GPT2LMHeadModel(GPT2Config()).to("cuda").half().state_dict(),GPT2LMHeadModel(GPT2Config()).to("cuda").half().state_dict()])
end=time.time()
print(f"Time: {end-start}")
start=time.time()
encodings=torch.randint(0,10,(1000,1000)).to("cuda").half()
for encoding in encodings:
    subStart=time.time()
    classifier.getExpert(encoding,"asdf")
    subEnd=time.time()
    print(f"Subtime: {subEnd-subStart}")
end=time.time()
print(f"Time: {end-start}")'''
from dataAnalysis import DataAnalysis
from memory import MemoryHandler
import numpy as np

class Classifier:
    def __init__(self, expertStructure, weights, threshold, expertMergeCount, cutThreshold):
        self.weights=weights
        self.threshold=threshold
        self.expertMergeCount=expertMergeCount
        self.cutThreshold=cutThreshold
    def getExpert(self, encoding):
        memory=MemoryHandler()
        experts=memory.read("*")
        comparisons={}
        for expertEncoding in experts:
            comparisons[expertEncoding]=np.dot(np.array(encoding)-np.array(expertEncoding),self.weights)
        for comparison,difference in comparisons:
            comparisonAvg=sum(comparisons.values())/len(comparisons.values())
            comparisons[comparison]=1-(difference-comparisonAvg)/comparisonAvg
        bestExpert=max(comparisons.values())
        if bestExpert<self.threshold:
            mergingExperts=[]
            mergingExperts.append(bestExpert)
            self.removeKeyFromDictionary(comparisons,bestExpert)
            for i in range(mergingExperts):
                maxComparison=max(comparisons.values())
                bestExpert=self.getKeyFromValue(comparisons,maxComparison)
                mergingExperts.append(bestExpert)
                self.removeKeyFromDictionary(comparisons,[bestExpert])
            checked=[]
            mergingExpertsFiltered=[]
            for i in range(len(bestExpert)):
                for j in range(len(bestExpert)):
                    difference=(abs(bestExpert[i]-bestExpert[j]))/min(bestExpert[i],bestExpert[j])
                    if [i,j] not in checked and [j,i] not in checked:
                        checked.append([i,j])
                        if difference>self.cutThreshold:
                            mergingExpertsFiltered.append(max([bestExpert[i],bestExpert[j]]))
            if len(mergingExpertsFiltered)==0:
                mergingExpertsFiltered=max(mergingExperts)
            newExpert=self.mergeExperts(mergingExpertsFiltered)
            memory.writeExpert(encoding, newExpert)
            return [encoding, newExpert]
        else:
            return [bestExpert,None]
    def getKeyFromValue(self, dictionary, desiredValue):
        for key,value in dictionary:
            if value==desiredValue:
                return key
    def removeKeyFromDictionary(self, dictionary, removedKey):
        returnDict={}
        for key, value in dictionary:
            if key not in removedKey:
                returnDict[key]=value
        return returnDict
    def mergeExperts(self, mergingExperts):
        if mergingExperts==None:
            return "Expert merging is none"
        else:
            #GRADMERGEIMPLEMENTATION
            pass
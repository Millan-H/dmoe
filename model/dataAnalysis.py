import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import cupy as cp
import numpy as np
import time
from memory import MemoryHandler
from datasets import load_dataset
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.decomposition import PCA
import gzip
import json
import regex as re
from classifier import Classifier
import ast

class DataAnalysis:
    def __init__(self,referenceStructure=None):
        self.referenceStructure=referenceStructure
        self.encodedReferences=[]
        self.analyzer=SentenceTransformer("all-MiniLM-L6-v2").to("cuda").half()
        self.memory=MemoryHandler()
        self.data=None
        self.encodedData=None
    def getEncoding(self, data, skipEncoding=False):
        if self.encodedReferences==[]:
            self.encodedReferences=self.memory.getEncodingStructure()
        if not skipEncoding:
            inputEncoded=torch.tensor(self.analyzer.encode(data),dtype=torch.bfloat16).to("cuda")
        else:
            inputEncoded=data
        encoding=f.cosine_similarity(
            torch.tensor(inputEncoded).unsqueeze(1), 
            torch.stack(self.encodedReferences).unsqueeze(0),
            dim=2
        )
        return encoding
    def getEncodingParallel(self, data):
        if self.encodedReferences==[]:
            self.encodedReferences=self.memory.getEncodingStructure()
            print(len(self.encodedReferences))
        data = f.normalize(data, p=2, dim=1)
        references = f.normalize(torch.stack(self.encodedReferences), p=2, dim=1)
        references.t()

        encoding = torch.mm(data, references.t())
        print(len(encoding))
        print(len(encoding[0]))
        return encoding
    def getEncodingLightweight(self, data):
        return self.analyzer.encode(data, show_progress_bar=False, convert_to_tensor=True).to("cuda").half()
    def getStructure(self):
        return self.referenceStructure
    def getEncodedReferences(self):
        return self.encodedReferences
    def getBaseModel(self):
        return self.analyzer
    def getEmbeddings(self,dataset=None,datapath="C:/Users/milla/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/239/arxiv-metadata-oai-snapshot.json",minTopicSize=150,sampleBatch=200000,topicTotal=1000000,topicBatch=500000,groupThreshold=0.98,restriction=None,base=True,classifier: Classifier=None):
        if base:
            start=time.time()
            pca=PCA(n_components=5, random_state=42)
            encoder=SentenceTransformer("all-MiniLM-L6-v2").to("cuda").half()
            topicClassifer=BERTopic(embedding_model=encoder)
            if dataset==None:
                data=[]
                with open(datapath) as file:
                    for line in file:
                        lineData=json.loads(line)
                        if re.search(r'\bcs\.|\bcs\b', lineData["categories"]):
                            data.append(lineData["abstract"])
            else:
                data=dataset
            subStart=time.time()
            if restriction==None:
                restriction=len(data)
            clusters=topicClassifer.fit_transform(data[0:restriction])
            subEnd=time.time()
            print(f"Clustering Time: {subEnd-subStart}")
            with open("C:/Users/milla/dmoe/model/encodingStructure.json",'w') as file:
                encodings=json.dump({"encodingStructure":topicClassifer.topic_embeddings_.tolist()[1:]},file)
            with open("C:/Users/milla/dmoe/model/encodings.json", "w") as file:
                clusters=json.dump({"clusters":clusters.tolist()}, file)
            end=time.time()
        else:
            start=time.time()
            pca=PCA(n_components=5, random_state=42)
            encoder=SentenceTransformer("all-MiniLM-L6-v2").to("cuda").half()
            topicClassifer=BERTopic(embedding_model=encoder)
            if dataset==None:
                data=[]
                with open(datapath) as file:
                    for line in file:
                        lineData=json.loads(line)
                        if re.search(r'\bcs\.|\bcs\b', lineData["categories"]):
                            data.append(lineData["abstract"])
            else:
                data=dataset
            print('asdfasdf')
            if restriction==None:
                restriction=len(data)
            encoded=encoder.encode(data[0:restriction])
            print("data",len(encoded))
            clusteringSet=[]
            for i in range(len(encoded)):
                clusteringSet.append(torch.tensor(encoded[i],dtype=torch.bfloat16).to("cuda"))
            clusteringSet=torch.stack(clusteringSet)
            clusteringSet=self.getEncodingParallel(clusteringSet)
            print('asdfasdf')
            hdbscan=HDBSCAN()
            umap=UMAP(n_components=65)
            umaped=umap.fit_transform(clusteringSet.tolist())
            print('asdfasdf')
            subStart=time.time()
            clustered=hdbscan.fit_predict(umaped)
            subEnd=time.time()
            print(f"Clustering Time: {subEnd-subStart}")
            with open("C:/Users/milla/dmoe/model/encodings.json", "w") as file:
                clusters=json.dump({"clusters":clustered.tolist()}, file)
            end=time.time()
            checked=[]
            averages=[]
            sorted={}
            cache=[]
            for i,cluster in enumerate(clustered):
                if cluster!=-1:
                    if cluster not in checked:
                        checked.append(cluster)
                        sorted[cluster]=[]
                    sorted[cluster].append(data[i])
                else:
                    cache.append(data[i])
            checked=[] 
            clustered=torch.tensor(clustered).sort()[0].tolist()
            for i,cluster in enumerate(clustered):
                if cluster!=-1:
                    if cluster not in checked:
                        checked.append(cluster)
                        averages.append([])
                    averages[checked.index(cluster)].append(torch.tensor(clusteringSet[i],dtype=torch.float32))
            averagesAveraged=[
                torch.mean(torch.stack(sublist), dim=0).tolist()
                for sublist in averages if sublist
            ]
            output={}
            for cluster,strings in sorted.items():
                output[f"{averagesAveraged[list(sorted.keys()).index(cluster)]}"]=strings
            classifier.setSortEncodings(list(map(ast.literal_eval,output.keys())))
            dependencies=[]
            for encoding in output.keys():
                print(type(list(ast.literal_eval(encoding))))
                dependencies.append(len(classifier.sort(list(ast.literal_eval(encoding)))))
            dependencies=torch.argsort(torch.tensor(dependencies)).tolist()
            sortedOutput={}
            for dependency in dependencies:
                sortedOutput[list(output.keys())[dependency]]=output[list(output.keys())[dependency]]
            sortedOutput["cache"]=cache
            return sortedOutput
    def clusterer(self, data, threshold=0.7):
        data=torch.tensor(data,dtype=torch.float32).to("cuda")
        clusters=torch.zeros(len(data), data.shape[1], dtype=torch.float32).to("cuda")
        sizes=torch.zeros(len(data), dtype=torch.float32).to("cuda")
        cache=torch.zeros(len(data), data.shape[1], dtype=torch.float32).to("cuda")
        cachedCount=0
        numClusters=0
        for datum in data:
            if numClusters==0:
                clusters[numClusters] = datum
                sizes[numClusters] = 1
                numClusters += 1
            else:
                similarity=f.cosine_similarity(
                    datum.unsqueeze(0), 
                    clusters[:numClusters],
                    dim=1
                )
                best=similarity.max()
                if best>threshold:
                    index=similarity.argmax()
                    clusters[index]=(clusters[index]*sizes[index]+datum)/(sizes[index]+1)
                    sizes[index]+=1
                else:
                    if best<1-2*(1-threshold):
                        clusters[numClusters] = datum
                        sizes[numClusters] = 1
                        numClusters += 1
                    else:
                        cache[cachedCount] = datum
                        cachedCount += 1
        if cachedCount > 0:
            for i in range(cachedCount):
                cached = cache[i]
                if numClusters==0:
                    clusters[numClusters] = cached
                    sizes[numClusters] = 1
                    numClusters += 1
                else:
                    similarity=f.cosine_similarity(
                        cached.unsqueeze(0), 
                        clusters[:numClusters],
                        dim=1
                    )
                    best=similarity.max()
                    if best>threshold:
                        index=similarity.argmax()
                        clusters[index]=(clusters[index]*sizes[index]+cached)/(sizes[index]+1)
                        sizes[index]+=1
                    else:
                        if best<1-2*(1-threshold):
                            clusters[numClusters] = cached
                            sizes[numClusters] = 1
                            numClusters += 1
        return clusters,sizes
    def getAverages(self,restriction=None,compareEmbeddings=False):
        clusters=[]
        sorted=[]
        checked=[]
        encoder=SentenceTransformer("all-MiniLM-L6-v2").to("cuda").half()
        abstracts=[]
        with open("C:/Users/milla/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/239/arxiv-metadata-oai-snapshot.json") as file:
            count=0
            for line in file:
                lineData=json.loads(line)
                if re.search(r'\bcs\.|\bcs\b', lineData["categories"]):
                    abstracts.append(lineData["abstract"])
                    if restriction!=None:
                        count+=1
                        if count>=restriction:
                            break
        self.data=abstracts
        encodings=encoder.encode(abstracts, show_progress_bar=True)
        self.encodedData=encodings
        with open("C:/Users/milla/dmoe/model/encodings.json", "r") as file:
            data=json.load(file)
            clusters=data["clusters"]
        for i,cluster in enumerate(clusters[0:10000]):
            if cluster not in checked and cluster>=0:
                checked.append(cluster)
                sorted.append([])
            if cluster>=0:
                sorted[checked.index(cluster)].append(encodings[i])
        averages=[torch.tensor(category, dtype=torch.float32).mean(dim=0) for category in sorted]
        return [averages,sorted]
    def sortOrder(self,encodings,classifier: Classifier):
        classifier.setSortEncodings(encodings)
        lengthsEncodings=[]
        lengthsList=[]
        for encoding in encodings:
            needed=classifier.sort(encoding).tolist()
            lengthsEncodings.append(encoding)
            lengthsList.append(len(needed))
        sortedLengths=torch.argsort(torch.tensor(lengthsList),descending=True).tolist()
        return sortedLengths
    def matchData(self, groupOrderIndecies, encodings, clusters):
        if self.encodedData.tolist()==None:
            if self.data==None:
                abstracts=[]
                encoder=SentenceTransformer("all-MiniLM-L6-v2").to("cuda").half()
                with open("C:/Users/milla/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/239/arxiv-metadata-oai-snapshot.json") as file:
                    for line in file:
                        lineData=json.loads(line)
                        if re.search(r'\bcs\.|\bcs\b', lineData["categories"]):
                            abstracts.append(lineData["abstract"])
            else:
                abstracts=self.data
            encodedData=encoder.encode(abstracts, show_progress_bar=True).tolist()
        else:
            encodedData=self.encodedData.tolist()
        def vectorIndex(vec,vecList):
            similarities = torch.cdist(
                vec.unsqueeze(0),
                torch.stack(vecList),
                p=2
            )
            return torch.nonzero(similarities == 0, as_tuple=False)
        matchedData={}
        for groupIndex in groupOrderIndecies:
            groupData=clusters[groupIndex]
            indecies=[]
            for datum in groupData:
                indecies.append(encodedData.index(datum.tolist()))
            matchedData[encodings[groupIndex]]=indecies
        return matchedData
    def pairText(self,matchedData,restriction=None):
        text=[]
        if self.data==None:
            with open("C:/Users/milla/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/239/arxiv-metadata-oai-snapshot.json") as file:
                count=0
                for line in file:
                    lineData=json.loads(line)
                    if re.search(r'\bcs\.|\bcs\b', lineData["categories"]):
                        text.append(lineData["abstract"])
                        if restriction!=None:
                            count+=1
                            if count>=restriction:
                                break
        else:
            text=self.data
        preparedDataset={}
        for encoding, data in matchedData.items():
            encodingDataset=[]
            for datum in data:
                encodingDataset.append(text[datum])
            preparedDataset[encoding]=encodingDataset
        return preparedDataset
    def preprocessData(self, dataPath="C:/Users/milla/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/239/arxiv-metadata-oai-snapshot.json", rawData=None, cluster=True, restriction=None):
        print("Started")
        if cluster==True:
            self.getEmbeddings(dataset=rawData,datapath=dataPath,restriction=restriction)
        start=time.time()
        averagesOutput=self.getAverages(restriction=restriction)
        encodings=averagesOutput[0]
        fracturedData=averagesOutput[1]
        end=time.time()
        print(f"Averaging and Sorting Time: {end-start}")
        start=time.time()
        trainingOrder=self.sortOrder(encodings, Classifier(0.95,0.75,7))
        end=time.time()
        print(f"Ordering Time: {end-start}")
        start=time.time()
        matchedData=self.matchData(trainingOrder, encodings, fracturedData)
        end=time.time()
        print(f"Dataifying Time: {end-start}")
        start=time.time()
        paired=self.pairText(matchedData,restriction=restriction)
        end=time.time()
        print(f"Pairing Time: {end-start}")
        return paired

dataAnalysis=DataAnalysis()
print(dataAnalysis.getEmbeddings(base=False,restriction=150,classifier=Classifier(4.8,5,4)))
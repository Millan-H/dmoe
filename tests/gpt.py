from controller import DMoE
import torch
from torch.nn import Sequential, Transformer, TransformerEncoderLayer, TransformerDecoderLayer, Linear, ReLU, Conv1d, Conv2d, Conv3d, MaxPool1d, MaxPool2d, MaxPool3d
from torch.optim import Adam

dataAnalyzers=[] #set up annotated the pile/c4/discord conversations
dataExperts=[] #set up the pile/c4/discord conversations
dmoe=DMoE(TransformerEncoderLayer(512,8), TransformerDecoderLayer(512,8),512,[1 for i in range(512)],0.8,10,0.3)
dmoe.trainAnalysis(dataAnalyzers)
dmoe.trainExperts(dataExperts)

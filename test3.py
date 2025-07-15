import torch
from torch.nn import MSELoss
from torch.optim import Adam
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, AutoModel

test=GPT2LMHeadModel(GPT2Config()).to("cuda").half()

data="asdfasdfasdfasdf asdf asdf asdf asdf asdf asdfasdfasdf asdfasdf asdfasd asdf asdfasdf asdfasdfasd asdfasdfasdf asdf asdfasdfasdfasdfasdf"
tokenizer=GPT2Tokenizer(GPT2Config()).to("cuda").half()
tokenized=tokenizer(data, return_tensors="pt", padding=True, truncation=True).to("cuda").half()
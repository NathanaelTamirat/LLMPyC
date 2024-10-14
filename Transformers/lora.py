#lowrank adaptation(Lora)
# Wo + BA
import torch 
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,in_features: int,out_features: int, bias: bool,r: int,alpha: int=None):
        super().__init__()

        if alpha is None: #Set α=r is not provided. i.e. make the scaling factor  α/r=1
            alpha=r     
        # freez the parameters
        self.weight=nn.Parameter(torch.empty((out_features,in_features))) # pretrained weight Wo=d*k
        self.weight.requires_grad=False 
        if bias:
            self.bias=nn.Parameter(torch.empty(out_features))
            self.bias.requires_grad=False
        else:
            self.bias=None

        self.scaling=alpha/r #scale factor 

        self.lora_a=nn.Parameter(torch.empty((r,in_features))) # a (r * k)
        self.lora_b=nn.Parameter(torch.empty((out_features,r))) # b (d * r) 
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a,a=5 ** 0.5) # init A 
            nn.init.zeros_(self.lora_b) # b=0 

    def forward(self,x: torch.Tensor):
        result=nn.functional.linear(x,self.weight,bias=self.bias) #compute x.WoT + bo
        result+=(x@self.lora_a.T@self.lora_b.T)*self.scaling # (alpha/r). x . AT . BT

class Embedding(nn.Module):
    def __init__(self,num_embeddings: int, embedding_dim: int, r: int, alpha:int=None):
        super().__init__()
        if alpha is None:
            alpha=r
        # pre trained embedding weights WoT (frozen)
        self.weight=nn.Parameter(torch.empty((num_embeddings,embedding_dim)))
        self.requires_grad_=False

        self.scaling=alpha/r #scale factor 
        self.lora_a=nn.Parameter(torch.empty((r,num_embeddings))) # a (r * k)
        self.lora_b=nn.Parameter(torch.empty((embedding_dim,r))) # b (d * r) 
        with torch.no_grad():
            nn.init.normal_(self.lora_a)
            nn.init.zeros_(self.lora_b)
    def forward(self,x: torch.Tensor):
        result=nn.functional.embedding(x,self.weight) #onehot(x) Wo​
        result+=(nn.functional.embedding(x,self.lora_a.T)@self.lora_b.T)*self.scaling #alpha/r (onehot(x).AT.BT)
        return result      
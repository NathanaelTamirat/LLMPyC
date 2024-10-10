import os 
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class GELU(nn.Module):
    def forward(self,input):
        return 0.5 * input *(1.0 +torch.tanh(math.sqrt(2.0/math.pi) * (input + 0.044715 * torch.pow(input,3.00))))

FLASH=0 #flash attention

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias",torch.trill(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))
        
    def forward(self,x):
        B,T,C=x.size() # batch,token,number of embedding dimension
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embed,dim=2)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #(B,nh,T,hs)      C // self.n_head is the size of each head's subspace.
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #(B,nh,T,hs)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #(B,nh,T,hs)
        if FLASH:
            y= F.scaled_dot_product_attention(q,k,v,is_causal=True)
        else:
            #MANUAL IMPLEMETATION OF ATTENTION
            att=(q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1))) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
            att=att.masked_fill(self.bias[:,:T,:T]==0,float('-inf'))
            att=F.softmax(att,dim=-1)
            y=att@v #(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=GELU()
        self.c_proj=nn.Linear(4* config.n_embd,config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG=1

    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()  
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

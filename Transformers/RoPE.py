import torch
import torch.nn as nn
import math
from typing import List,Optional

class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self,d_model: int, heads:int, d_k:int,bias:bool=True):
        super().__init()
        self.linear=nn.Linear(d_model,heads*d_k,bias=bias)
        self.heads=heads
        self.d_k=d_k
        
    def forward(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]
        head_shape=x.shape[:-1] #apply head to last dim
        x=self.linear(x)
        x=x.view(*head_shape,self.heads,-1,self.d_k) #[seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,heads: int,d_model: int,dropout_prob: float=0.1,bias: bool=True ):
        super().__init__()
        self.heads=heads
        self.d_k=d_model//heads
        self.query=PrepareForMultiHeadAttention(d_model,heads,self.d_k,bias=bias)
        self.key=PrepareForMultiHeadAttention(d_model,heads,self.d_k,bias=bias)
        self.value=PrepareForMultiHeadAttention(d_model,heads,self.d_k,bias=True)
        self.softmax=nn.Softmax(dim=1) # Softmax for attention along the time dimension of `key`
        self.dropout=nn.Dropout(dropout_prob)
        self.output=nn.Linear(d_model,d_model)
        self.scale=1/math.sqrt(self.d_k) # Scaling factor before the softmax
        self.attn=None # for storing attention weights
    
    def get_scores(self,query: torch.Tensor,key: torch.Tensor):
        return torch.einsum('ibhd,jbhd->ijbh',query,key)
    
    def prepare_mask(self,mask: torch.Tensor,query_shape: List[int], key_shape: List[int]):
        #input mask has shape `[seq_len, seq_len, batch_size]
        assert mask.shape[0]==1 or mask.shape[0]==query_shape[0]
        assert mask.shape[1]==key_shape[0]
        assert mask.shape[2]==1 or mask.shape[2]==query_shape[1]
        mask=mask.unsqueeze(-1)
        return mask #[seq_len_q, seq_len_k, batch_size, heads]
    
    def forward(self,*,query: torch.Tensor,key: torch.Tensor,value: torch.Tensor, mask: Optional[torch.Tensor]=None):
        seq_len,batch_size,_=query.shape
        if mask is not None:
            mask=self.prepare_mask(mask,query.shape,key.shape)
        
        query=self.query(query)
        key=self.key(key)
        value=self.value(value)
        scores=self.get_scores(query,key)
        scores*=self.scale

        if mask is not None:
            scores=scores.masked_fill(mask==0,float('-inf'))
        
        attn=self.softmax(scores)
        attn = self.dropout(attn)
        x=torch.einsum('ijbh,jbhd->ibhd',attn,value)
        self.attn=attn.detach() #save attention weights
        x=x.reshape(seq_len,batch_size,-1)  # Concatenate multiple heads
        return self.output(x)
    

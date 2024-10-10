import math
import copy
from typing import Optional, List
import torch 
import torch.nn as nn
import torch.nn.functional as F

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

class FeedForward(nn.Module):
    def __init__(self,d_model: int, 
                 d_ff: int,
                 dropout: float=0.1,
                 activation=nn.ReLU(),
                 is_gated: bool=False,
                 bias1: bool=True,
                 bias2: bool=True,
                 bias_gate: bool=True
                 ):
        super().__init__()
        self.layer1=nn.Linear(d_model,d_ff,bias=bias1)
        self.layer2=nn.Linear(d_ff,d_model,bias=bias2)
        self.dropout=nn.Dropout(dropout)
        self.activation=activation
        self.is_gated=is_gated
        if is_gated:
            self.linear_v=nn.Linear(d_model,d_ff,bias=bias_gate)

    def forward(self,x:torch.Tensor):
        g=self.activation(self.layer1(x))
        if self.is_gated:
            x=g*self.linear_v(x)
        else:
            x=g
        x=self.dropout(x)
        x=self.layer2(x)
        return x


#https://medium.com/image-processing-with-python/positional-encoding-in-the-transformer-model-e8e9979df57f#:~:text=The%20Transformer%20embeds%20positional%20information,a%20token%20within%20a%20sentence.
def get_positional_encoding(d_model: int,max_len: int=5000):
    encodings=torch.zeros(max_len,d_model) # empty encoding matrix
    position=torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1) #position index
    two_i=torch.arange(0,d_model,2,dtype=torch.float32) #2i for sin and 2i+1 for cos
    div_term=torch.exp(two_i*-(math.log(10000.0)/d_model)) 
    encodings[:,0::2]=torch.sin(position*div_term)
    encodings[:,1::2]=torch.cos(position*div_term)
    encodings=encodings.unsqueeze(1).requires_grad_(False)
    return encodings

'''
 input tensor x is often structured as a 2D 
 tensor with shape (batch_size, sequence_length).
'''

#embedding tokens and add fixed positinal encodeing
class Embedding_with_positional_encoding(nn.Module):
    def __init__(self,d_model: int,n_vocab: int,max_len: int=5000):
        super().__init__()
        self.linear=nn.Embedding(n_vocab,d_model)
        self.d_model=d_model
        self.register_buffer("positional_encoding",get_positional_encoding(d_model,max_len))
    def forward(self,x: torch.Tensor):
        pe=self.positional_encodings[:x.shape[1]].requires_grad_(False)
        return self.linear(x)*math.sqrt(self.d_model)+pe #By multiplying the output of the embedding layer by math.sqrt(self.d_model), you ensure that the variance of the embeddings is approximately 1. This helps in stabilizing the training process and can lead to faster convergence.
   
#embedding token and add trainable positional encoding
class Embedding_with_trainable_positional_encoding(nn.Module):
    def __init__(self,d_model: int,n_vocab: int, max_len: int=5000):
        super().__init__()
        self.linear=nn.Embedding(n_vocab,d_model)
        self.d_model=d_model
        self.positional_encodings=nn.Parameter(torch.zeros(max_len,d_model),requires_grad_=True)
    def forward(self,x: torch.Tensor):
        pe=self.positional_encodings[:x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model)+pe

#transformer layer
class Transformer_layer(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention = None,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        super().__init__()
        self.d_model=d_model
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.dropout=nn.Dropout(dropout_prob)
        self.norm_self_attn=nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn=nn.LayerNorm([d_model])
        self.norm_ff=nn.LayerNorm([d_model])
        self.is_save_ff_input=False #save feed forward input

    def forward(self,*,
                x: torch.Tensor,
                mask: torch.Tensor,
                src: torch.Tensor=None,
                src_mask: torch.Tensor=None):
        z=self.norm_self_attn(x) # Normalize the vectors before doing self attention
        self_attn=self.self_attn(query=z,key=z,value=z,mask=mask) #self attention
        x=x+self.dropout(self_attn) #Add the self attention results to the input
        if src is not None:
            z=self.norm_src_attn(x)
            src_attn=self.src_attn(query=z,key=src,value=src,mask=src_mask)
            x=x+self.dropout(src_attn)
        
        z=self.norm_ff(x) # Normalize the vectors before doing feed forward
        if self.is_save_ff_input:
            self.ff_input=z.clone()
        ff=self.feed_forward(z)
        x=x+self.dropout(ff)
        return x    

class Encoder(nn.Module):
    def __init__(self,layer: Transformer_layer,n_layers: int):
        super().__init__()
        self.layers=nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)]) #clone transformer layer
        self.norm=nn.LayerNorm([layer.size])

    def forward(self,x:torch.Tensor,mask:torch.Tensor):
        for layer in self.layers:
            x=layer(x=x,mask=mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, layer: Transformer_layer, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        return self.norm(x)

class Generator(nn.Module):
    '''
    This predicts the tokens and gives the lof softmax of those.
    You don't need this if you are using `nn.CrossEntropyLoss`.
    '''
    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        return self.projection(x)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        enc = self.encode(src, src_mask)
        return self.decode(enc, src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
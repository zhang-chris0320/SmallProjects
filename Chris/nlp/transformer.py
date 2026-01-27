import torch
from torch import nn
import torch.nn.functional as F
import math
import torch.nn as nn
from torch import tensor
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocb_size,embedding_dim):   #两个参数，一个词汇表大小，一个embedding维度
        super(TokenEmbedding,self).__init__(vocb_size,embedding_dim,padding_idx=1)
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim,max_len,device):
        super(PositionalEmbedding,self).__init__()
        self.encoding=torch.zeros(max_len,embedding_dim,device=device)   #初始化0矩阵
        self.encoding.requires_grad_=False
        pos=torch.arange(0,max_len,device=device )
        pos=pos.float().unsqueeze(dim=1) #这行hyw
        _zi=torch.arange(0,embedding_dim,step=2,device=device).float()
        self.encoding[:,0::2]=torch.sin(pos/(10000**(pos/embedding_dim)))
        self.encoding[:,1::2]=torch.cos(pos/(10000**(pos/embedding_dim)))
    def forward(self,x):
        batch_size,seq_len=x.size()
        return self.encoding[:seq_len,:]
class TransformerEmbedding(nn.Module):
    def __init__(self,vocb_size,embedding_dim,max_len,drop_prob,device):
        super().__init__()
        self.tok_emb=TokenEmbedding(vocb_size,embedding_dim)
        self.pos_emb=PositionalEmbedding(embedding_dim,max_len,device)
        self.drop_out=nn.Dropout(p=drop_prob)
    def forward(self,x):
        tok_emb=self.tok_emb(x)
        pos_emb=self.pos_emb(x)
        return self.drop_out

class SelfAttention(nn.Module):
    def __init__(self, dim_q,dim_k,dim_v):
        super(SelfAttention,self).__init__()
        self.dim_q=dim_q
        self.dim_k=dim_k
        self.dim_v=dim_v
        self.linear_q=nn.Linear(dim_q,dim_k,bias=False)
        self.linear_k=nn.Linear(dim_q,dim_k,bias=False)
        self.linear_v=nn.Linear(dim_q,dim_v,bias=False)
        self._norm_fact=1/math.sqrt(dim_k)
    def forward(self,x):
        batch,n,dim_q=x.shape
        assert dim_q== self.dim_q
        q=self.linear_q(x)
        k=self.linear_k(x)
        v=self.linear_v(x)
        dist=torch.bmm(q,k.transpose(1,2))*self._norm_fact
        dist=F.softmax(dist,dim=-1)
        att=torch.bmm(dist,v)
        return att
class  MutiheadAttention(nn.Module): 
    def __init__(self, embedding_dim,n_head):
        super(MutiheadAttention,self).__init__()
        self.n_head=n_head
        self.embedding_dim=embedding_dim
        self.k_linear=nn.Linear(embedding_dim,embedding_dim)
        self.q_linear=nn.Linear(embedding_dim,embedding_dim)
        self.v_linear=nn.Linear(embedding_dim,embedding_dim)
        self.d_token=embedding_dim
        self.out=nn.Linear(embedding_dim,embedding_dim)

    def foward(self,q,k,v):
        batch=q.size(0)
        k=self.k_linear(k).view(batch,-1,self.n_head,self.d_token)
        q=self.q_linear(q).view(batch,-1,self.n_head,self.d_token)
        v=self.v_linear(v).view(batch,-1,self.n_head,self.d_token) 
        k=k.transpose(1,2)
        q=q.transpose(1,2)
        v=v.transpose(1,2)
        scores=self.attentiton(q,k,v,self.d_token)
        scores=scores.transpose(1,2).contiguous().view(batch,-1,self.embedding_dim)
        output=self.out(scores)
        return output
    @staticmethod
    def attention(q,k,v,d_token):
        scores=torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(d_token)
        attn=F.softmax(scores,dim=-1)
        output=torch.matmul(attn,v)
        return output
class LayerNorm(nn.Module):
    def __init__(self, embedding_dim,eps=1e-12):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(embedding_dim))
        self.beta=nn.Parameter(torch.zero(embedding_dim))
        self.eps=eps
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        var=x.var(-1,unbiased=False,keepdim=True)
        out=(x-mean)/torch.sqrt(var+self.eps)
        out=self.gamma*out+self.beta
        return out
class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim,hidden,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.fc1=nn.Linear(embedding_dim,hidden)
        self.fc2=nn.Linear(hidden,embedding_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x

class Encoderlayer(nn.Module):
    def __init__(self, embedding_dim,ffn_hidden,n_head,dropout=0.1):
        super(Encoderlayer).__init__()
        self.attention=MutiheadAttention(embedding_dim,n_head)
        self.norm1=LayerNorm(embedding_dim)
        self.dropout1=nn.Dropout(dropout)
        self.ffn=PositionwiseFeedForward(embedding_dim,ffn_hidden,dropout)
        self.norm2=LayerNorm(embedding_dim)
        self.dropout2=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        _x=x
        x=self.attention(x,x,x,mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)
        _x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        return x
class Encoder(nn.Module):
    def __init__(self,enc_voc_size,max_len, embedding_dim,ffn_hidden,n_head,n_layer,dropout=0.1):
        super(Encoder,self).__init__()
        self.embedding=TransformerEmbedding(enc_voc_size,max_len, embedding_dim,dropout=0.1)
        self.layers=nn.ModuleList (
             [
              Encoderlayer(embedding_dim,ffn_hidden,n_head)
              for _ in range(n_layer)



             ]
        )
    def forward(self,x,s_mask):
        x=self.embedding(x)
        for layer in self.layers:
            X=layer(x,s_mask)
        return x     
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim,ffn_hidden,n_head,drop_prob):
        super(DecoderLayer,self).__init__()
        self.attentiin1=MutiheadAttention(embedding_dim,n_head)
        self.norm1=LayerNorm(embedding_dim)
        self.dropout1=nn.Dropout(drop_prob)
        self.cross_attetion=MutiheadAttention(embedding_dim,n_head)
        self.norm2=LayerNorm(embedding_dim)
        self.dropout2=nn.Dropout(drop_prob)
        self.ffn=PositionwiseFeedForward(embedding_dim,ffn_hidden,drop_prob)
        self.norm3=LayerNorm(embedding_dim)
        self.dropout3=nn.Dropout(drop_prob)
    def forward(self,dec,enc,t_mask,s_mask):
        _x=dec
        x=self.attention1(dec,dec,dec,t_mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)
        _x=x
        x=self.cross_attention(x,enc,enc,s_mask)
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        x=self.ffn(x)
        x=self.dropout3(x)
        x=self.norm(x+_x)
        return x
class Decoder(nn.Module):
    def __init__(self, dec_voc_size,max_len,embedding_dim,ffn_hidden,n_head,n_layer,drop_prob,device):
        super(Decoder,self).__init__()
        self.embedding=TransformerEmbedding(dec_voc_size,max_len, embedding_dim,dropout=0.1)
        self.layers=nn.ModuleList (
             [
              DecoderLayer(embedding_dim,ffn_hidden,n_head)
              for _ in range(n_layer)



             ]
        
        )
        self.fc=nn.linear(embedding_dim,dec_voc_size)
    def forward(self,dec,enc,t_mask,s_mask):
        dec=self.embedding(enc)
        for layer in self.layers:
            dec=layer(dec,enc,t_mask,s_mask)
        dec=self.fc(dec)
        return dec                        

  
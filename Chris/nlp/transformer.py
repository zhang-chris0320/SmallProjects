import torch
from torch import nn
import torch.nn.functional as F
import math
import torch.nn as nn
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from pathlib import Path
from collections import Counter
import pandas as pd
from tokenizer import EnglishTokenizer,ChineseTokenizer



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
class Transformer(nn.Module):
    def __init__(self, 
                src_pad_ix,
                trg_pad_ix,
                enc_voc_size,
                dec_voc_size,
                embedding_dim,
                max_len,
                n_heads ,
                ffn_hidden,
                n_layers,
                drop_prob,
                device 
                 ):
        super(Transformer,self).__init__()
        self.encoder=Encoder(

            enc_voc_size,
            max_len,
            embedding_dim,
            ffn_hidden,
            n_heads,
            n_layers,
            drop_prob,
            device,
        )
        self.decoder=Decoder(dec_voc_size,max_len,embedding_dim,ffn_hidden,n_heads,n_layers,drop_prob,device)
        self.src_pad_idx=src_pad_ix
        self.trg_pad_idx=trg_pad_ix
        self.device=device
    def make_pad_mask(self,q,k,pad_idx_q,pad_idx_k):
        len_q,len_k=q.size(),k.size(1)
        q=q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q=q.repeat(1,1,1,len_k)
        k=k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k=k.repeat(1,1,len_q,1)
        mask=q&k
        return mask
    def make_casual_mask(self,q,k):
        mask=torch.tril(torch.ones(q,k)).type(torch.BoolTensor).to(self.device)
        return mask
    def forward(self,src,trg):
        src_mask=self.make_pad_mask(src,src,self.src_pad_idx,self.src_pad_idx)
        trg_mask=self.make_pad_mask(trg,trg,self.trg_pad_idx,self.trg_pad_idx)*self.make_casual_mask(trg,trg)
        enc=self.encoder(src,src_mask)
        out=self.decoder(trg,src,trg_mask,src_mask)
        return out

class ChineseTokenizer:
    @staticmethod
    def build_vocab(texts, save_path, vocab_size=10000):
        words = []
        for text in texts:
            words.extend(jieba.cut(text))
        counter = Counter(words).most_common(vocab_size - 4)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('<pad>\n<sos>\n<eos>\n<unk>\n')
            for w, _ in counter:
                f.write(w + '\n')
    
    @staticmethod
    def from_vocab(path):
        t = ChineseTokenizer()
        t.vocab = {}
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                t.vocab[line.strip()] = i
        return t
    
    def encode(self, text, add_sos_eos=False):
        ids = [self.vocab.get(w, self.vocab['<unk>']) for w in jieba.cut(text)]
        if add_sos_eos:
            ids = [self.vocab['<sos>']] + ids + [self.vocab['<eos>']]
        return ids

class EnglishTokenizer:
    @staticmethod
    def build_vocab(texts, save_path, vocab_size=10000):
        words = []
        for text in texts:
            words.extend(text.lower().split())
        counter = Counter(words).most_common(vocab_size - 4)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('<pad>\n<sos>\n<eos>\n<unk>\n')
            for w, _ in counter:
                f.write(w + '\n')
    
    @staticmethod
    def from_vocab(path):
        t = EnglishTokenizer()
        t.vocab = {}
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                t.vocab[line.strip()] = i
        return t
    
    def encode(self, text, add_sos_eos=False):
        ids = [self.vocab.get(w, self.vocab['<unk>']) for w in text.lower().split()]
        if add_sos_eos:
            ids = [self.vocab['<sos>']] + ids + [self.vocab['<eos>']]
        return ids

def process():
    Path('models').mkdir(exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    train = pd.read_json('D:/translation2019zh/translation2019zh_train.json', lines=True)
    test = pd.read_json('D:/translation2019zh/translation2019zh_valid.json', lines=True)
    
    train = train[['english', 'chinese']].rename(columns={'english':'en', 'chinese':'zh'})
    test = test[['english', 'chinese']].rename(columns={'english':'en', 'chinese':'zh'})
    
    ChineseTokenizer.build_vocab(train['zh'].tolist(), 'models/zh_vocab.txt')
    EnglishTokenizer.build_vocab(train['en'].tolist(), 'models/en_vocab.txt')
    
    zh_tk = ChineseTokenizer.from_vocab('models/zh_vocab.txt')
    en_tk = EnglishTokenizer.from_vocab('models/en_vocab.txt')
    
    train['zh'] = train['zh'].apply(lambda x: zh_tk.encode(x, False))
    train['en'] = train['en'].apply(lambda x: en_tk.encode(x, True))
    train.to_json('data/processed/train.jsonl', orient='records', lines=True, force_ascii=False)
    
    test['zh'] = test['zh'].apply(lambda x: zh_tk.encode(x, False))
    test['en'] = test['en'].apply(lambda x: en_tk.encode(x, True))
    test.to_json('data/processed/test.jsonl', orient='records', lines=True, force_ascii=False)
    
    print("完成")

class Dataset(Dataset):
    def __init__(self, path, max_len=50):
        self.data = [json.loads(l) for l in open(path, encoding='utf-8')]
        self.max_len = max_len
    
    def __getitem__(self, i):
        src = self.data[i]['en'][:self.max_len]
        trg = self.data[i]['zh'][:self.max_len]
        if len(src) < self.max_len:
            src += [0] * (self.max_len - len(src))
        if len(trg) < self.max_len:
            trg += [0] * (self.max_len - len(trg))
        return torch.tensor(src), torch.tensor(trg)
    
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    process()
   


def train_one_epoch(model,dataloader,loss_fn,optimizer,device):
    total_loss=0
    model.train()
    for inputs, targets in tqdm(dataloader,desc='训练'):
        encoder_inputs=inputs.to(device)
        decoder_inputs=targets[:,:-1]
        decoder_targets=targets[:,1:]
        encoder_outputs,context_vector=model.encoder(encoder_inputs)
        decoder_hidden=context_vector.unsqueeze(0)
        decoder_outputs=[]
        seq_len =decoder_inputs.shape[1]
        for i in range(seq_len):
            decoder_input=decoder_inputs[:,1].unsequeeze(1)
            decoder_output,decoder_hidden=model.decoder(decoder_input,decoder_hidden,encoder_outputs)
            decoder_outputs.append(decoder_output)
        decoder_outputs=torch.cat(decoder_outputs,dim=1)
        decoder_outputs=decoder_outputs.reshape(-1,decoder_outputs.shape[-1])
        decoder_targets=decoder_targets.reshape(-1)
        loss= loss_fn(decoder_outputs,decoder_targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss/ len(dataloader)

def train():
    device=torch.device('cuda' if torch.cuda.is_available()else 'cpu')
    
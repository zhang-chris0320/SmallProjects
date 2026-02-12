import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import jieba

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbedding, self).__init__(vocab_size, embedding_dim, padding_idx=0)

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, embedding_dim, device=device)
        self.encoding.requires_grad_ = False
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, device=device).float() * -(math.log(10000.0) / embedding_dim))
        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
    
    def forward(self, x):
        return self.encoding[:x.size(1), :].unsqueeze(0).repeat(x.size(0), 1, 1)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len, drop_prob, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_emb = PositionalEmbedding(embedding_dim, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        return self.drop_out(self.tok_emb(x) + self.pos_emb(x))

class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(MultiheadAttention, self).__init__()
        assert embedding_dim % n_head == 0
        self.n_head = n_head
        self.embedding_dim = embedding_dim
        self.d_k = embedding_dim // n_head
        
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        
        q = self.w_q(q).view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.embedding_dim)
        return self.fc(out)

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * out + self.beta

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden)
        self.fc2 = nn.Linear(hidden, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, ffn_hidden, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(embedding_dim, n_head)
        self.norm1 = LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(embedding_dim, ffn_hidden, dropout)
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(x))
        _x = x
        x = self.ffn(x)
        x = self.norm2(x + self.dropout2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, embedding_dim, ffn_hidden, n_head, n_layer, dropout, device):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, embedding_dim, max_len, dropout, device)
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, ffn_hidden, n_head, dropout)
            for _ in range(n_layer)
        ])
    
    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, ffn_hidden, n_head, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(embedding_dim, n_head)
        self.norm1 = LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.cross_attention = MultiheadAttention(embedding_dim, n_head)
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffn = PositionwiseFeedForward(embedding_dim, ffn_hidden, dropout)
        self.norm3 = LayerNorm(embedding_dim)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.self_attention(dec, dec, dec, t_mask)
        x = self.norm1(x + self.dropout1(x))
        
        _x = x
        x = self.cross_attention(x, enc, enc, s_mask)
        x = self.norm2(x + self.dropout2(x))
        
        _x = x
        x = self.ffn(x)
        x = self.norm3(x + self.dropout3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, embedding_dim, ffn_hidden, n_head, n_layer, dropout, device):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, embedding_dim, max_len, dropout, device)
        self.layers = nn.ModuleList([
            DecoderLayer(embedding_dim, ffn_hidden, n_head, dropout)
            for _ in range(n_layer)
        ])
        self.fc = nn.Linear(embedding_dim, dec_voc_size)
    
    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        return self.fc(dec)

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, 
                 embedding_dim, max_len, n_heads, ffn_hidden, n_layers, drop_prob, device):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        self.encoder = Encoder(enc_voc_size, max_len, embedding_dim, ffn_hidden, n_heads, n_layers, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, embedding_dim, ffn_hidden, n_heads, n_layers, drop_prob, device)
    
    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        q = (q != self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        k = (k != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        return q & k
    
    def make_causal_mask(self, size):
        mask = torch.tril(torch.ones(size, size)).bool().to(self.device)
        return mask
    
    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src)
        trg_pad_mask = self.make_pad_mask(trg, trg)
        trg_len = trg.size(1)
        trg_causal_mask = self.make_causal_mask(trg_len)
        trg_mask = trg_pad_mask & trg_causal_mask.unsqueeze(0).unsqueeze(1)
        
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

class ChineseTokenizer:
    @staticmethod
    def build_vocab(texts, save_path, vocab_size=10000):
        words = []
        for text in texts[:10000]:
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
        for text in texts[:10000]:
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
    
    train = pd.read_json('D:/translation2019zh/translation2019zh_train.json', lines=True, nrows=100000)
    valid = pd.read_json('D:/translation2019zh/translation2019zh_valid.json', lines=True, nrows=10000)
    
    train = train[['english', 'chinese']].rename(columns={'english':'en', 'chinese':'zh'})
    valid = valid[['english', 'chinese']].rename(columns={'english':'en', 'chinese':'zh'})
    
    ChineseTokenizer.build_vocab(train['zh'].tolist(), 'models/zh_vocab.txt')
    EnglishTokenizer.build_vocab(train['en'].tolist(), 'models/en_vocab.txt')
    
    zh_tk = ChineseTokenizer.from_vocab('models/zh_vocab.txt')
    en_tk = EnglishTokenizer.from_vocab('models/en_vocab.txt')
    
    train['zh'] = train['zh'].apply(lambda x: zh_tk.encode(x, False))
    train['en'] = train['en'].apply(lambda x: en_tk.encode(x, True))
    train.to_json('data/processed/train.jsonl', orient='records', lines=True, force_ascii=False)
    
    valid['zh'] = valid['zh'].apply(lambda x: zh_tk.encode(x, False))
    valid['en'] = valid['en'].apply(lambda x: en_tk.encode(x, True))
    valid.to_json('data/processed/valid.jsonl', orient='records', lines=True, force_ascii=False)
    
    print("数据处理完成")

class TranslationDataset(Dataset):
    def __init__(self, path, max_len=50):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
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

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    
    for src, trg in tqdm(dataloader, desc='训练'):
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg[:, :-1])
        
        output = output.reshape(-1, output.shape[-1])
        target = trg[:, 1:].reshape(-1)
        
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

if __name__ == '__main__':
    process()
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = TranslationDataset('data/processed/train.jsonl', max_len=50)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    zh_tk = ChineseTokenizer.from_vocab('models/zh_vocab.txt')
    en_tk = EnglishTokenizer.from_vocab('models/en_vocab.txt')
    
    model = Transformer(
        src_pad_idx=0,
        trg_pad_idx=0,
        enc_voc_size=len(en_tk.vocab),
        dec_voc_size=len(zh_tk.vocab),
        embedding_dim=256,
        max_len=50,
        n_heads=8,
        ffn_hidden=512,
        n_layers=3,
        drop_prob=0.1,
        device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(10):
        loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
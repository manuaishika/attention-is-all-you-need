"""
Full Implementation of "Attention Is All You Need" Paper
Transformer Model for English → Hindi/Maithili Translation

This is a complete, production-ready implementation following the original paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy
import time
import sentencepiece as spm
import os
import sys
import traceback
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Python < 3.7 fallback
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# Model hyperparameters (paper defaults)
N = 6                    # Number of encoder/decoder layers
d_model = 512           # Model dimension
d_ff = 2048             # Feed-forward dimension
h = 8                   # Number of attention heads
dropout = 0.1           # Dropout rate
max_length = 100        # Maximum sequence length
batch_size = 64         # Batch size for training
num_epochs = 20         # Number of training epochs
learning_rate = 0.0001  # Learning rate

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
    return ~subsequent_mask

# ============================================================================
# ATTENTION MECHANISM
# ============================================================================

def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """Implements Figure 2 from the paper"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# ============================================================================
# POSITION-WISE FEED-FORWARD NETWORKS
# ============================================================================

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# ============================================================================
# EMBEDDINGS AND POSITIONAL ENCODING
# ============================================================================

class Embeddings(nn.Module):
    """Token embeddings multiplied by sqrt(d_model)"""
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """Implement the PE function from the paper."""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ============================================================================
# LAYER NORMALIZATION AND RESIDUAL CONNECTIONS
# ============================================================================

class LayerNorm(nn.Module):
    """Construct a layernorm module."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))

# ============================================================================
# ENCODER
# ============================================================================

class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (Figure 1, left)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# ============================================================================
# DECODER
# ============================================================================

class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (Figure 1, right)"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# ============================================================================
# GENERATOR (Output Layer)
# ============================================================================

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

# ============================================================================
# FULL TRANSFORMER MODEL
# ============================================================================

class Transformer(nn.Module):
    """The full Transformer model"""
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model

# ============================================================================
# BATCH HANDLING
# ============================================================================

class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, tgt=None, pad=0):
        self.src = src.to(device)
        self.src_mask = (self.src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1].to(device)
            self.tgt_y = tgt[:, 1:].to(device)
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).to(device)
        return tgt_mask

# ============================================================================
# DATASET AND VOCABULARY
# ============================================================================

class SentencePieceVocab:
    """Wrapper for SentencePiece tokenizer"""
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        self.sp.load(model_path)
    
    def encode(self, text):
        return self.sp.encode_as_ids(text)
    
    def decode(self, ids):
        if isinstance(ids, list):
            # Filter out special tokens for cleaner output
            ids = [id for id in ids if id not in [self.bos_id(), self.eos_id(), self.pad_id(), self.unk_id()]]
        return self.sp.decode_ids(ids)
    
    def bos_id(self):
        return self.sp.bos_id()
    
    def eos_id(self):
        return self.sp.eos_id()
    
    def pad_id(self):
        return self.sp.pad_id()
    
    def unk_id(self):
        return self.sp.unk_id()
    
    def get_vocab_size(self):
        return self.sp.get_piece_size()

class TranslationDataset(Dataset):
    """Dataset for translation pairs"""
    def __init__(self, pairs, src_vocab, tgt_vocab, max_length=100):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        
        # Encode with BOS and EOS tokens
        src_tokens = self.src_vocab.encode(src_text)
        tgt_tokens = self.tgt_vocab.encode(tgt_text)
        
        # Truncate if too long
        src_tokens = src_tokens[:self.max_length]
        tgt_tokens = tgt_tokens[:self.max_length]
        
        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)

def collate_fn(batch):
    """Collate function for DataLoader"""
    src_batch, tgt_batch = [], []
    for src, tgt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)
    
    # Pad sequences
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_batch, tgt_batch

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class LabelSmoothing(nn.Module):
    """Implement label smoothing from the paper."""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

class SimpleLossCompute:
    """A simple loss compute and train function."""
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return loss.data * norm

def run_epoch(data_iter, model, loss_compute):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    model.train()
    
    for i, batch in enumerate(data_iter):
        batch = Batch(batch[0], batch[1], pad=0)
        out = model.forward(batch.src, batch.tgt, 
                            batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        
        if i % 50 == 1:
            elapsed = time.time() - start
            print(f"  Step {i} Loss: {loss / batch.ntokens:.6f} "
                  f"Tokens/sec: {tokens / elapsed:.1f}")
            start = time.time()
            tokens = 0
    
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

# ============================================================================
# INFERENCE
# ============================================================================

def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_symbol=3):
    """Greedy decoding for inference"""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)
    
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           ys,
                           subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        # Stop if we generate EOS token
        if next_word == eos_symbol:
            break
    
    return ys

def translate(model, src_vocab, tgt_vocab, sentence, max_len=50):
    """Translate a sentence from source to target language"""
    model.eval()
    src_tokens = src_vocab.encode(sentence)
    src = torch.tensor([src_tokens], device=device)
    src_mask = (src != 0).unsqueeze(-2)
    
    with torch.no_grad():
        result = greedy_decode(model, src, src_mask, max_len=max_len, 
                               start_symbol=tgt_vocab.bos_id(),
                               eos_symbol=tgt_vocab.eos_id())
    
    translation = tgt_vocab.decode(result[0].tolist())
    return translation

# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_sample_data():
    """Create sample English-Hindi-Maithili data for demonstration"""
    sample_pairs = [
        # English to Hindi - Greetings
        ("hello how are you", "नमस्ते आप कैसे हैं"),
        ("i am fine", "मैं ठीक हूँ"),
        ("good morning", "शुभ प्रभात"),
        ("good afternoon", "नमस्कार"),
        ("good evening", "शुभ संध्या"),
        ("good night", "शुभ रात्रि"),
        ("nice to meet you", "आपसे मिलकर खुशी हुई"),
        ("see you later", "बाद में मिलते हैं"),
        ("see you tomorrow", "कल मिलते हैं"),
        ("have a nice day", "आपका दिन शुभ हो"),
        
        # English to Hindi - Personal Information
        ("what is your name", "आपका नाम क्या है"),
        ("my name is john", "मेरा नाम जॉन है"),
        ("i am a student", "मैं एक छात्र हूँ"),
        ("i am a teacher", "मैं एक शिक्षक हूँ"),
        ("how old are you", "आपकी उम्र क्या है"),
        ("i am twenty years old", "मैं बीस साल का हूँ"),
        ("where do you live", "आप कहाँ रहते हैं"),
        ("i live in delhi", "मैं दिल्ली में रहता हूँ"),
        ("where are you from", "आप कहाँ से हैं"),
        ("i am from mumbai", "मैं मुंबई से हूँ"),
        
        # English to Hindi - Questions
        ("what time is it", "क्या समय हुआ है"),
        ("what is this", "यह क्या है"),
        ("who are you", "आप कौन हैं"),
        ("where is the bathroom", "बाथरूम कहाँ है"),
        ("where is the hospital", "अस्पताल कहाँ है"),
        ("where is the school", "स्कूल कहाँ है"),
        ("how much does this cost", "यह कितने का है"),
        ("what do you want", "आप क्या चाहते हैं"),
        ("how can i help you", "मैं आपकी कैसे मदद कर सकता हूँ"),
        ("what are you doing", "आप क्या कर रहे हैं"),
        
        # English to Hindi - Actions and Activities
        ("i am learning python", "मैं पाइथन सीख रहा हूँ"),
        ("i love programming", "मुझे प्रोग्रामिंग पसंद है"),
        ("i am reading a book", "मैं एक किताब पढ़ रहा हूँ"),
        ("i am watching television", "मैं टेलीविजन देख रहा हूँ"),
        ("i am going to school", "मैं स्कूल जा रहा हूँ"),
        ("i am eating food", "मैं खाना खा रहा हूँ"),
        ("i am drinking water", "मैं पानी पी रहा हूँ"),
        ("i am sleeping", "मैं सो रहा हूँ"),
        ("i am working", "मैं काम कर रहा हूँ"),
        ("i am studying", "मैं पढ़ाई कर रहा हूँ"),
        
        # English to Hindi - Feelings and States
        ("i am hungry", "मुझे भूख लगी है"),
        ("i am thirsty", "मुझे प्यास लगी है"),
        ("i am tired", "मैं थक गया हूँ"),
        ("i am happy", "मैं खुश हूँ"),
        ("i am sad", "मैं दुखी हूँ"),
        ("i am angry", "मैं गुस्सा हूँ"),
        ("i am excited", "मैं उत्साहित हूँ"),
        ("i am worried", "मैं चिंतित हूँ"),
        ("i am sorry", "मुझे माफ करें"),
        ("i am happy to see you", "आपको देखकर खुशी हुई"),
        
        # English to Hindi - Common Phrases
        ("thank you very much", "बहुत बहुत धन्यवाद"),
        ("thank you", "धन्यवाद"),
        ("please help me", "कृपया मेरी मदद करें"),
        ("please come in", "कृपया अंदर आएं"),
        ("please sit down", "कृपया बैठिए"),
        ("excuse me", "माफ करें"),
        ("i do not understand", "मैं समझा नहीं"),
        ("can you repeat that", "क्या आप दोहरा सकते हैं"),
        ("speak slowly please", "कृपया धीरे बोलें"),
        ("yes i understand", "हाँ मैं समझ गया"),
        
        # English to Hindi - Numbers and Time
        ("one two three", "एक दो तीन"),
        ("what is the date today", "आज क्या तारीख है"),
        ("it is monday today", "आज सोमवार है"),
        ("it is morning", "सुबह है"),
        ("it is afternoon", "दोपहर है"),
        ("it is evening", "शाम है"),
        ("it is night", "रात है"),
        
        # English to Hindi - Family and Relationships
        ("this is my father", "यह मेरे पिता हैं"),
        ("this is my mother", "यह मेरी माँ हैं"),
        ("i have a brother", "मेरा एक भाई है"),
        ("i have a sister", "मेरी एक बहन है"),
        ("how is your family", "आपका परिवार कैसा है"),
        ("my family is good", "मेरा परिवार ठीक है"),
        
        # English to Hindi - Food and Drinks
        ("i want to eat", "मुझे खाना चाहिए"),
        ("i like rice", "मुझे चावल पसंद है"),
        ("i like tea", "मुझे चाय पसंद है"),
        ("i like coffee", "मुझे कॉफी पसंद है"),
        ("i want water", "मुझे पानी चाहिए"),
        ("the food is delicious", "खाना स्वादिष्ट है"),
        
        # English to Maithili - Greetings
        ("hello how are you", "प्रणाम अहां केहन छी"),
        ("i am fine", "हम ठीक छी"),
        ("good morning", "शुभ प्रभात"),
        ("good night", "शुभ रात्रि"),
        ("nice to meet you", "अहांक संग भेट मे खुशी भेल"),
        ("see you later", "बाद मे मिलू"),
        
        # English to Maithili - Personal Information
        ("what is your name", "अहांक नाम की अछि"),
        ("my name is john", "हमर नाम जॉन अछि"),
        ("i am a student", "हम छात्र छी"),
        ("i am a teacher", "हम शिक्षक छी"),
        ("how old are you", "अहांक उम्र की अछि"),
        ("where do you live", "अहां कतय रहैत छी"),
        ("i live in delhi", "हम दिल्ली मे रहैत छी"),
        ("where are you from", "अहां कतयक संग छी"),
        
        # English to Maithili - Questions
        ("what time is it", "किएक समय भेल अछि"),
        ("what is this", "ई की अछि"),
        ("where is the bathroom", "बाथरूम कतय अछि"),
        ("what do you want", "अहां की चाहीत छी"),
        ("how can i help you", "हम अहांक कतेक मदद करि सकैत छी"),
        
        # English to Maithili - Actions
        ("i am learning python", "हम पाइथन सीखैत छी"),
        ("i love programming", "हमरा प्रोग्रामिंग पसंद अछि"),
        ("i am reading a book", "हम किताब पढ़ैत छी"),
        ("i am eating food", "हम खाना खा रहल छी"),
        ("i am drinking water", "हम पानी पी रहल छी"),
        ("i am going to school", "हम स्कूल जा रहल छी"),
        ("i am working", "हम काम करैत छी"),
        
        # English to Maithili - Feelings
        ("i am hungry", "हमरा भूख लागल अछि"),
        ("i am thirsty", "हमरा प्यास लागल अछि"),
        ("i am tired", "हम थकल छी"),
        ("i am happy", "हम खुश छी"),
        ("i am sad", "हम दुखी छी"),
        ("i am sorry", "हमरा माफ करब"),
        
        # English to Maithili - Common Phrases
        ("thank you very much", "बहुत बहुत धन्यवाद"),
        ("thank you", "धन्यवाद"),
        ("please help me", "कृपया हमर मदद करू"),
        ("please come in", "कृपया भीतर आउ"),
        ("excuse me", "माफ करू"),
        ("i do not understand", "हम नहि समझल"),
        ("yes i understand", "हाँ हम समझ गेल"),
        
        # English to Maithili - Food
        ("i want to eat", "हमरा खाना चाही"),
        ("i like rice", "हमरा चावल पसंद अछि"),
        ("i like tea", "हमरा चाय पसंद अछि"),
        ("i want water", "हमरा पानी चाही"),
    ]
    return sample_pairs

def train_tokenizer(pairs, vocab_size=8000, model_prefix="spm_model"):
    """Train SentencePiece tokenizer on the dataset"""
    print("Training SentencePiece tokenizer...")
    
    # Write all text to a file for tokenizer training
    training_file = f"{model_prefix}_training.txt"
    with open(training_file, "w", encoding="utf-8") as f:
        for eng, trans in pairs:
            f.write(f"{eng}\n{trans}\n")
    
    # Train SentencePiece model
    spm.SentencePieceTrainer.Train(
        f'--input={training_file} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--character_coverage=1.0 '
        f'--pad_id=0 '
        f'--unk_id=1 '
        f'--bos_id=2 '
        f'--eos_id=3 '
        f'--model_type=bpe '
        f'--max_sentence_length=10000'
    )
    
    print(f"OK: Tokenizer trained and saved as {model_prefix}.model")
    return f"{model_prefix}.model"

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function"""
    print("=" * 70)
    print("FULL TRANSFORMER IMPLEMENTATION - Attention Is All You Need")
    print("=" * 70)
    
    try:
        # Step 1: Create or load dataset
        print("\nStep 1: Preparing dataset...")
        train_pairs = create_sample_data()
        print(f"OK: Created {len(train_pairs)} translation pairs")
        
        # Step 2: Train tokenizer
        print("\nStep 2: Training tokenizer...")
        tokenizer_model_path = train_tokenizer(train_pairs, vocab_size=2000, model_prefix="spm_model")
        
        # Step 3: Initialize vocabularies
        print("\nStep 3: Initializing vocabularies...")
        SRC = SentencePieceVocab(tokenizer_model_path)
        TGT = SentencePieceVocab(tokenizer_model_path)
        vocab_size = SRC.get_vocab_size()
        print(f"OK: Vocabulary size: {vocab_size}")
        
        # Step 4: Create dataset and dataloader
        print("\nStep 4: Creating dataloader...")
        dataset = TranslationDataset(train_pairs, SRC, TGT, max_length=max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        print(f"OK: Dataset created with {len(dataset)} samples")
        
        # Step 5: Initialize model
        print("\nStep 5: Initializing Transformer model...")
        print(f"   Model parameters: N={N}, d_model={d_model}, d_ff={d_ff}, h={h}")
        model = make_model(
            vocab_size,
            vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"OK: Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        # Step 6: Setup training
        print("\nStep 6: Setting up training...")
        criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        print(f"OK: Optimizer: Adam (lr={learning_rate})")
        
        # Step 7: Training loop
        print("\nStep 7: Starting training...")
        print(f"   Training for {num_epochs} epochs...")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            loss_compute = SimpleLossCompute(model.generator, criterion, optimizer)
            avg_loss = run_epoch(dataloader, model, loss_compute)
            print(f"Average Loss: {avg_loss:.6f}")
        
        print("\n" + "=" * 70)
        print("Training completed!")
        
        # Step 8: Save model
        print("\nStep 8: Saving model...")
        model_path = "transformer_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'src_vocab_size': vocab_size,
            'tgt_vocab_size': vocab_size,
            'N': N,
            'd_model': d_model,
            'd_ff': d_ff,
            'h': h,
            'dropout': dropout,
        }, model_path)
        print(f"OK: Model saved to {model_path}")
        
        # Step 9: Test translations
        print("\nStep 9: Testing translations...")
        print("=" * 70)
        
        test_sentences = [
            "hello my friend",
            "what is your name",
            "i am learning ai",
            "good morning everyone",
            "thank you very much"
        ]
        
        for sent in test_sentences:
            translation = translate(model, SRC, TGT, sent)
            print(f"English: {sent}")
            try:
                print(f"Translation: {translation}")
            except UnicodeEncodeError:
                # Fallback for Windows console encoding issues
                print(f"Translation: [Hindi/Maithili text - check output file]")
            print("-" * 70)
        
        print("\n" + "=" * 70)
        print("SUCCESS! Your Transformer is trained and ready!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nStack trace:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


import torch
import torch.nn as nn

class DenoisingTransformer(nn.Module):
    def __init__(self, vocab, max_seq_length, d_model=256, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.token_emb = nn.Embedding(vocab.VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(max_seq_length, d_model)
        self.time_emb = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.output_head = nn.Linear(d_model, 20)  # Seulement les 20 AA
        
    def forward(self, x, t):
        B, L = x.shape
        
        # Embeddings
        h = self.token_emb(x)
        h += self.pos_emb(torch.arange(L, device=x.device)).unsqueeze(0)
        h += self.time_emb(t.unsqueeze(1)).expand(-1, L, -1)
        
        src_key_padding_mask = (x == self.vocab.PAD_TOKEN)
        
        # Transformer
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        
        # Sortie
        return self.output_head(h)
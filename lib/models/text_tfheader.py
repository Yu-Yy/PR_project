import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=2,num_heads = 2):
        super(TextEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=64)
        # norm = nn.LayerNorm(embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
        self.positional_encodings = nn.Parameter(torch.rand(1500, embedding_dim), requires_grad=True)

    def forward(self,seq_embed):

        seq_embed = seq_embed + self.positional_encodings[:seq_embed.shape[1], :] #.T #.unsqueeze(0) # B * S * E(2)
        seq_embed = seq_embed.permute(1, 0, 2) # S N E
        # import pdb; pdb.set_trace()
        encode_feature = self.transformer_encoder(seq_embed)
        encode_feature = encode_feature.permute(1,0,2) # N S E
        extracted_feature = encode_feature.squeeze(-1)
        return extracted_feature 
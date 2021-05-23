import torch
import torch.nn as nn

class VitEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=16, embedding_dim=64,num_heads = 4):
        super(VitEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=256)
        norm = nn.LayerNorm(embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4, norm=norm)
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)
        self.seqlize = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self,f):
        seq_embed = self.seqlize(f).flatten(2)
        seq_embed = seq_embed + self.positional_encodings[:seq_embed.shape[2], :].T.unsqueeze(0)
        # import pdb;pdb.set_trace()
        seq_embed = seq_embed.permute(2,0,1) # S N E
        encode_feature = self.transformer_encoder(seq_embed)
        encode_feature = encode_feature.permute(1,0,2)
        extracted_feature = nn.functional.adaptive_avg_pool1d(encode_feature,1)
        extracted_feature = extracted_feature.squeeze(-1)


        return extracted_feature 
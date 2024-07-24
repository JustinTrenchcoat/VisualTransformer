import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1,1,embed_dim)), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(size=(1,num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,x):
        x = 2
        return x
        
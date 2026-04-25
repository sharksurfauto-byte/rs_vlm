import torch
import torch.nn as nn

class PatchEmbedFromFeatureMap(nn.Module):
    """
    convert the feature map from the cnn to token seq
    [B,256,28,28] -> [B,784,384]
    """

    def __init__(self, in_channels:int=256, embed_dim:int=384):
        super().__init__()
        #linear proj from CNN channel dim to ViT embed dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self,x:torch.Tensor) -> tuple[torch.Tensor,int,int]:
        # x shape: [B,256,28,28]
        x = self.proj(x) #[b,384,28,28]
        B,C,H,W = x.shape
        x=x.flatten(2) #[b,384,784]
        x=x.transpose(1,2) #[b,784,384]
        return x,H,W
    
class ViTBlock(nn.Module):
    """
    Single transformer block: LayerNorm -> Attention -> LayerNorm -> MLP
    """
    def __init__(self, embed_dim:int =384, num_heads:int=6,
                 mlp_ratio:float=4.0, dropout:float=0.0):
        super().__init__()
        self.norm1=nn.LayerNorm(embed_dim)
        self.attn=nn.MultiheadAttention(
            embed_dim,num_heads,dropout=dropout, batch_first=True
            )
        self.norm2=nn.LayerNorm(embed_dim)
        mlp_hidden=int(embed_dim*mlp_ratio)
        self.mlp=nn.Sequential(
            nn.Linear(embed_dim,mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden,embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # self attn with residual
        normed=self.norm1(x)
        attn_out,_=self.attn(normed, normed, normed)
        x = self.norm2(x)
        #MLP with residual
        x = x + self.mlp(x)
        return x
    
class ViTBody(nn.Module):
    """
    lightweight ViT body thats opertyed on CNN feat map
    """
    #x:[B,256,28,28]  input form cnn stem
    #output shape: [b,784,384] tokens seq for proj

    def __init__(
            self,
            in_channels:int=256,
            embed_dim: int = 384,
            num_heads: int = 6,
            num_blocks: int = 4,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedFromFeatureMap(in_channels, embed_dim)
        #learnable CLS token
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        #pos embedding: 784 pathc tokens +1 cls token
        self.pos_embed=nn.Parameter(torch.zeros(1,785,embed_dim))  #why 785? 784+1cls token

        self.blocks=nn.ModuleList(
            [ViTBlock(embed_dim,num_heads,mlp_ratio,dropout) 
             for _ in range(num_blocks)]
        )

        self.norm=nn.LayerNorm(embed_dim)
        self.embed_dim=embed_dim

        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token,std=0.02)
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
    
    def forward(self, feature_map:torch.Tensor) -> torch.Tensor:
        # feature map shape: [B,256,28,28]
        #output tokens shape: [B,785,384] #cls + patch tokens

        B=feature_map.shape[0]
        #embed the patches
        x,H,W=self.patch_embed(feature_map) #[B,784,384]

        #prepenf CLS token
        cls=self.cls_token.expand(B,-1,-1) #[B,1,384]
        x=torch.cat((cls,x),dim=1) #[B,785,384]

        #add pos embed
        x = x + self.pos_embed #[B,785,384]

        #transformer blocks
        for block in self.blocks:
            x=block(x) #[B,785,384]
        x = self.norm(x) 

        return x 
    
#testing thie ViT body
# if __name__ == "__main__":
#     import sys
#     import os
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     from encoder.cnn_stem import CNNStem

#     stem = CNNStem(pretrained=False)
#     vit = ViTBody()

#     dummy = torch.randn(2, 3, 224, 224)
#     feature_map = stem(dummy)
#     print(f"CNN output:  {feature_map.shape}")

#     tokens = vit(feature_map)
#     print(f"ViT output:  {tokens.shape}")
#     assert tokens.shape == (2, 785, 384), "Shape mismatch!"
#     print("ViTBody shape check passed.")
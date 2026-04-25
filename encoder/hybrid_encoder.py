import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder.cnn_stem import CNNStem
from encoder.vit_body import ViTBody
from encoder.gsd_adapter import GSDAdapter

class HybridEncoder(nn.Module):
    """
    Pipeline:
        image -> CNNStem -> feature map
               -> GSDAdapter -> positional bias
               -> ViTBody (feature map + bias) -> token sequence

    Input:
        images: [B, 3, 224, 224]
        gsd:    [B]  — metres/pixel per image (default 10.0 for Sentinel-2)

    Output:
        tokens: [B, 785, 384]  — CLS + 784 patch tokens
    """

    def __init__(
            self,
            cnn_pretrained:bool = True,
            cnn_frozen:bool = False,
            embed_dim:int = 384,
            num_heads: int = 6,
            num_blocks: int = 4,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.cnn_stem=CNNStem(pretrained=cnn_pretrained,frozen=cnn_frozen)

        self.vit_body=ViTBody(
            in_channels=self.cnn_stem.out_channels, # this iwll be 256 for resnet-18 layer 3 o/p
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        self.gsd_adapter=GSDAdapter(embed_dim=embed_dim)

        self.embed_dim=embed_dim

    def forward(self,images:torch.Tensor, gsd:torch.Tensor) -> torch.Tensor:
        #images shape: [B,3,224,224]
        #gsd shape: [B] - GSD in m/pixel
        #output tokens shape: [B,785,384]

        B = images.shape[0]

        if gsd is None:
            gsd=torch.full((B,), 10.0, device=images.device) 
        
        #stage1: cnn stem extracts local feats
        feature_map=self.cnn_stem(images) #[B,256,28,28]

        #stage2: gsd adapter produces scale aware pos bias
        gsd_bias=self.gsd_adapter(gsd) #[B,1,384]

        #stage3: vit body: inject GSD bias into pos embeds
        original_pos_embed=self.vit_body.pos_embed #[1,785,384]
        self.vit_body.pos_embed=nn.Parameter(
            original_pos_embed + gsd_bias # [B,785,384]
        )

        tokens=self.vit_body(feature_map) #[B,785,384]
        return tokens
    
    def get_param_count(self):
        total=sum(p.numel() for p in self.parameters())
        trainable=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    

if __name__ == "__main__":
    encoder = HybridEncoder(cnn_pretrained=False)

    images = torch.randn(2, 3, 224, 224)
    gsd = torch.tensor([10.0, 0.3])   # two different resolutions

    tokens = encoder(images, gsd)
    print(f"Image input:  {images.shape}")
    print(f"GSD input:    {gsd}")
    print(f"Token output: {tokens.shape}")
    assert tokens.shape == (2, 785, 384), "Shape mismatch!"

    total, trainable = encoder.get_param_count()
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print("HybridEncoder shape check passed.")
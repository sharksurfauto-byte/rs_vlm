import torch
import torch.nn as nn

class GSDAdapter(nn.Module):
    """
    this is s scale_aaware pos bias from ground sample distance
    this takes a GSD scalar (m/pixel) and produces a 384 dim bias that gets added to the vit pos embeds
    """

    #x shape: [B] - GSD value per img in m/pixel
    #output: [B,1,384] -  bias to add to pos embed

    def __init__(self, embed_dim:int=384, hidden_dim:int = 64):
        super().__init__()
        
        self.mlp=nn.Sequential(
            nn.Linear(1,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,embed_dim)
        )

        self._init_weights()

    def _init_weights(self):
        #init as near 0 so the adapter starts as a no op
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self,gsd:torch.Tensor) -> torch.Tensor:
        #gsd shape: [B] - GSD in m/picel (eg 10.0 for sentinel-2)
        # returns bias shape :[B,1,384] this is added to the pos embeds of the vit

        #normalize gsd to a resonabel range
        gsd=torch.log(gsd).unsqueeze(-1) #[B,1]
        bias=self.mlp(gsd) #[B,384]
        bias=bias.unsqueeze(1) #[B,1,384]
        return bias

# testing the shapes lol  
# if __name__ == "__main__":
#     adapter = GSDAdapter()

#     # Simulate a batch of 2 images with different GSDs
#     # Sentinel-2 = 10m/px, commercial satellite = 0.3m/px
#     gsd = torch.tensor([10.0, 0.3])

#     bias = adapter(gsd)
#     print(f"GSD input:     {gsd}")
#     print(f"Bias output:   {bias.shape}")
#     assert bias.shape == (2, 1, 384), "Shape mismatch!"
#     print("GSDAdapter shape check passed.")

#     # Verify the two biases are different (model sees scale difference)
#     diff = (bias[0] - bias[1]).abs().mean().item()
#     print(f"Mean bias diff between GSDs: {diff:.4f} (should be > 0)")
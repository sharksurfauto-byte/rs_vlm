import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MLPProjector(nn.Module):
    """
    encoder ouptuts 384 dim vectors.. but LLM needs a 2048 dim vecotr. SO the MLP projects 384->2048
    """
    #X shape : [B,785,384]
    #ouput shape: [B,785,2048]

    def __init__(self, 
                 enc_dim:int=384, 
                 llm_dim:int=2048, 
                 hidden_dim:int=1024, 
                 dropout:float=0.1):
        super().__init__()

        self.mlp=nn.Sequential(
            nn.Linear(enc_dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,llm_dim)
        )
        self.norm=nn.LayerNorm(llm_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight,std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, tokens:torch.Tensor) -> torch.Tensor:
        #tokens shape: [B,785,384]
        #output shape: [B,785,2048]

        x=self.mlp(tokens) #[B,785,2048]
        x=self.norm(x)
        return x
    
    def get_param_count(self):
        total=sum(p.numel() for p in self.parameters())
        return total
    
#testing the MLP projector
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from encoder.hybrid_encoder import HybridEncoder

    # Full pipeline test: image -> encoder -> projector
    encoder = HybridEncoder(cnn_pretrained=False)
    projector = MLPProjector(enc_dim=384, llm_dim=2048)

    images = torch.randn(2, 3, 224, 224)
    gsd = torch.tensor([10.0, 0.3])

    tokens = encoder(images, gsd)           # [B, 785, 384]
    projected = projector(tokens)           # [B, 785, 2048]

    print(f"Encoder output:   {tokens.shape}")
    print(f"Projector output: {projected.shape}")
    assert projected.shape == (2, 785, 2048), "Shape mismatch!"

    print(f"Projector params: {projector.get_param_count():,}")
    print("MLPProjector shape check passed.")
# CNN stem
"""
What cnn stem does: It takes a raw satellite image tensor and outputs a feature map where every location denotes a meaningfuil local pattern...like edges, texttures, shapes..
"""

# input shape: [B,3,224,224] (b,c,h,w)
# outuput shape: [B,256,28,28] (feature map, 1/8 resolution)

"""
we use ResNet18, but not the fulll arch, we throw away the last 2 layers (the global avg and the classification head)- we only wanat the conv feat extractor, not the classifier
"""

import torch
import torch.nn as nn
import timm

class CNNStem(nn.Module):
    def __init__(self, pretrained:bool = True, frozen:bool = False):
        super().__init__()

        #load full resnet with pretrained imagenet weights
        resnet = timm.create_model("resnet18",pretrained=pretrained)

        # resnet18 stages: stem -> layer1 -> layer2 -> layer3 -> layer4 ...... westop at layer 3 (output channels=256, spatial=28x28 for 224 input)
        self.stem = nn.Sequential(
            resnet.conv1,  # type: ignore       #224->112
            resnet.bn1,  # type: ignore         
            resnet.act1,  # type: ignore
            resnet.maxpool,  # type: ignore     #112->56
            resnet.layer1,  # type: ignore      #56->56 (no stride) 
            resnet.layer2,  # type: ignore      #56->28 (stride=2)
            resnet.layer3,  # type: ignore      #28->28 
        )
        for module in self.stem[-1].modules():
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                module.stride = (1, 1)
                # NOTE: no dilation — the 1x1 downsample shortcut doesn't support it cleanly
                # simple stride removal keeps both main path and shortcut at 28x28

        # self.out_channels=256 #layer 3 outputs channels in Resnet-18

        # instead of hardcoding thhe output_channels directly, we use a more dynamic approach for it
        with torch.no_grad():
            dummy=torch.zeros(1,3,224,224) #dummy tensor of an image
            self.out_channels=self.stem(dummy).shape[1]

        if frozen:
            for param in self.stem.parameters():
                param.requires_grad = False

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # x shape: [B,3,H,W]
        #feature map shape: [B,256,H/8,W/8]
        return self.stem(x)
    
# if __name__ == "__main__":
#     #test the cnn stem
#     model = CNNStem(pretrained=False,frozen=False)
#     dummy = torch.randn(2,3,224,224) #batch of 2 images
#     out = model(dummy)
#     print(out.shape) #[2,256,28,28]
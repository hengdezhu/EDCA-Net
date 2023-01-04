import math
import torch.nn as nn

from torch.nn import Module

class eca_layer(Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, gamma = 2, b = 1):
        super(eca_layer, self).__init__()        
        
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, channel, kernel_size=channel, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).unsqueeze(-1)
        
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
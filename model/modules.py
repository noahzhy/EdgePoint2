import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicLayer(nn.Module):
    """
      Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
      return self.layer(x)


class ResNetLayer(nn.Module):
    
    def __init__(self, dim1, dim2):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(dim1, dim2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim2),
            nn.ReLU(True),
            nn.Conv2d(dim2, dim2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim2)
        )
        
        self.skip = nn.Identity() if dim1 == dim2 else nn.Conv2d(dim1, dim2, 1)
        
    def forward(self, x: torch.Tensor):
        return F.relu(self.layer(x) + self.skip(x))


class FasterNetLayer(nn.Module):
    
    def __init__(self, dim, n_div, n_mul):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim//n_div, dim//n_div, 3, 1, 1)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim*n_mul, 1, bias=False),
            nn.BatchNorm2d(dim*n_mul),
            nn.ReLU(True),
            nn.Conv2d(dim*n_mul, dim, 1)
        )
        
        self.dim1 = dim // n_div
        
    def forward(self, x: torch.Tensor):
        _x = torch.cat([
            self.conv1(x[:,:self.dim1]),
            x[:,self.dim1:]
        ], 1)
        return x + self.conv2(_x)
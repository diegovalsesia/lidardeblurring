import torch
import torch.nn as nn 
from .network import *
class QuadraticConnectionUnit(nn.Module):
    def __init__(self, channels):
        super(QuadraticConnectionUnit, self).__init__()
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, dep, x):
        return dep * x + self.bias
    

class QuadraticConnectionUnitSigmoid(nn.Module):
    def __init__(self, channels):
        super(QuadraticConnectionUnitSigmoid, self).__init__()
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, dep, x):
        return x * self.sigmoid(dep) + self.bias
    

class DepthPromptMoudle(nn.Module):
    def __init__(self, channels):
        super(DepthPromptMoudle, self).__init__()
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))
        self.sigmoid = nn.Sigmoid()
        self.transformer = TransformerBlock(dim=int(channels), num_heads=1,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias'   ## Other option 'BiasFree'
          )
        self.conv_out = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, dep, x):
        dep_prompt = x * self.sigmoid(dep) + self.bias
        x =torch.cat((dep_prompt, x), dim=1)
        x = self.conv_out(x)
        x = self.transformer(x)
        return x
    



class DepthTransformDecoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthTransformDecoder, self).__init__()
        self.conv1 = nn.Conv2d( in_channel, out_channel, 5, 1, 2)
        self.relu = nn.PReLU(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel * 4, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channel, out_channel * 4, 3, 1, 1)
        self.bias = nn.Parameter(torch.randn((1, out_channel * 4, 1, 1)))
        self.pixelshuffle = nn.PixelShuffle(2)
        self.layernorm = LayerNorm(out_channel * 4, 'WithBias')
        self.layernorm2 = LayerNorm(out_channel, 'WithBias')

    def forward(self, x):
        x_2 = self.conv3(x)
        x_2 = self.layernorm(x_2)
        x = self.conv1(x)
        x = self.layernorm2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x_2 * x + self.bias
        return  self.pixelshuffle(x)
    

class DepthTransformEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthTransformEncoder, self).__init__()
        self.conv1 = nn.Conv2d( in_channel, out_channel, 5, 1, 2)
        self.relu = nn.PReLU(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel //4, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channel, out_channel //4, 3, 1, 1)
        self.bias = nn.Parameter(torch.randn((1, out_channel // 4, 1, 1)))
        self.pixelunshuffle = nn.PixelUnshuffle(2)
        self.layernorm = LayerNorm(out_channel //4, 'WithBias')
        self.layernorm2 = LayerNorm(out_channel, 'WithBias')

    def forward(self, x):
        x_2 = self.conv3(x)
        x_2 = self.layernorm(x_2)
        x = self.conv1(x)
        x = self.layernorm2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x_2 * x + self.bias
        return  self.pixelunshuffle(x)
    

class EmbedDepthTransform(nn.Module):
    def __init__(self, out_channel):
        super(EmbedDepthTransform, self).__init__()
        self.conv1 = nn.Conv2d( 1, out_channel, 5, 1, 2)
        self.relu = nn.PReLU(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.layernorm = LayerNorm(out_channel, 'WithBias')
        self.layernorm2 = LayerNorm(out_channel, 'WithBias')
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layernorm(x)
        return  self.layernorm2(self.conv2(x))

####layer norm#####        
    

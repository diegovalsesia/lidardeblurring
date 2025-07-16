import torch.nn as nn
from .restormer import Restormer
import torch
from .srx8 import SYESRX8NetS
import torch.nn.functional as F
class CatRestormer(nn.Module):
    def __init__(self, 
        inp_channels=4, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):
        super(CatRestormer, self).__init__()

        ckpt = torch.load('./pretrainedModel/srx8.pkl', map_location='cuda')
        self.up_net = SYESRX8NetS(36)
        self.up_net.load_state_dict(ckpt)
        for param in self.up_net.parameters():
            param.requires_grad = False

        self.resotrmer = Restormer(
            inp_channels=inp_channels, 
            out_channels=out_channels, 
            dim = dim,
            num_blocks = num_blocks, 
            num_refinement_blocks = num_refinement_blocks,
            heads = heads,
            ffn_expansion_factor = ffn_expansion_factor,
            bias = bias,
            LayerNorm_type = LayerNorm_type,   ## Other option 'BiasFree'
            dual_pixel_task = dual_pixel_task)
    def forward(self,x,dep):
        dep =self.up_net(dep)
        b,c,h,w = x.size() 
        dep = F.interpolate(dep, size=(h,w))
        #dep = F.interpolate(dep, size=(h,w))
        inp = torch.cat((x,dep),dim=1)
        return self.resotrmer(inp)
    
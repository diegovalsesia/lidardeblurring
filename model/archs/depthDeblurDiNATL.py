from .DeblurDiNATL import *
from ..utils.adapter import  EmbedDepthTransform, DepthTransformDecoder, DepthPromptMoudle
from .srx8 import SYESRX8NetS
class Embeddings_depth_output(nn.Module):
    def __init__(self, dim, num_blocks, num_refinement_blocks, heads, bias):
        super(Embeddings_depth_output, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)
        
        self.de_trans_level3 = nn.Sequential(*[
            item for sublist in 
            [[TransBlock(dim*2**2, heads[2], 7, 1, 1, bias=bias),
             TransBlock(dim*2**2, heads[2], 7, 4, 1, bias=bias)] for i in range(num_blocks[2]//2)] for item in sublist
        ])
        
        self.up3_2 = nn.Sequential(
            nn.ConvTranspose2d(dim*2**2, dim*2, kernel_size=4, stride=2, padding=1, bias=bias),
            self.activation,
        )

        self.fusion_level2 = LGFF(dim*4, dim*2, 1, bias)

        self.de_trans_level2 = nn.Sequential(*[
            item for sublist in 
            [[TransBlock(dim*2, heads[1], 7, 1, 1, bias=bias),
             TransBlock(dim*2, heads[1], 7, 8, 1, bias=bias)] for i in range(num_blocks[1]//2)] for item in sublist
        ])

        self.up2_1 = nn.Sequential(
            nn.ConvTranspose2d(dim*2, dim, kernel_size=4, stride=2, padding=1, bias=bias),
            self.activation,
        )

        self.fusion_level1 = LGFF(dim*2, dim*1, 1, bias)


        self.de_trans_level1 = nn.Sequential(*[
            item for sublist in 
            [[TransBlock(dim, heads[0], 7, 1, 1, bias=bias),
             TransBlock(dim, heads[0], 7, 16, 1, bias=bias)] for i in range(num_blocks[0]//2)] for item in sublist
        ])
        
        self.refinement = nn.Sequential(*[
            item for sublist in 
            [[TransBlock(dim, heads[0], 7, 1, 1, bias=bias),
             TransBlock(dim, heads[0], 7, 16, 1, bias=bias)] for i in range(num_blocks[0]//2)] for item in sublist
        ])
        self.output = nn.Sequential(
            nn.Conv2d(dim, 3, kernel_size=3, padding=1, bias=bias),
            self.activation
        )

        ckpt = torch.load('./pretrainedModel/srx8.pkl', map_location='cuda')
        self.up_net = SYESRX8NetS(36)
        self.up_net.load_state_dict(ckpt)
        for param in self.up_net.parameters():
            param.requires_grad = False

        self.adapter_connect3_2 = DepthPromptMoudle(dim*2**1)
        self.adapter_connect2_1 = DepthPromptMoudle(dim)
        self.adapter_emb_dep =EmbedDepthTransform(dim*2**1)
        self.adapter_dep2_1 = DepthTransformDecoder(dim*2**1, dim)
    def forward(self, x, depth, residual_1, residual_2):
        depth = self.up_net(depth)
        depth_ten = self.adapter_emb_dep(depth)
        hx = self.de_trans_level3(x)
        hx = self.up3_2(hx)
        b,c,h,w = hx.size()
        depth_ten = F.interpolate(depth_ten,size=(h,w))
        hx = self.fusion_level2(torch.cat((hx, residual_2), dim=1))
        hx = self.adapter_connect3_2(depth_ten,hx)
        hx = self.de_trans_level2(hx)
        hx = self.up2_1(hx)
        hx = self.fusion_level1(torch.cat((hx, residual_1), dim=1))
        depth_ten = self.adapter_dep2_1(depth_ten)
        hx = self.adapter_connect2_1(depth_ten,hx)
        hx = self.de_trans_level1(hx)
        hx = self.refinement(hx)
        hx = self.output(hx)
        return hx


class DepthNADeblurL(nn.Module):
    def __init__(self,
                dim = 64, 
                num_blocks = [4,6,8],
                num_refinement_blocks = 4,
                num_heads = [2,4,8], 
                kernel = 7, 
                dilation = 3, 
                ffn_expansion_factor = 1, 
                bias = False):

        super(DepthNADeblurL, self).__init__()
        
        self.encoder = Embeddings(dim)

        self.multi_scale_fusion_level1 = LGFF(dim*7, dim*1, ffn_expansion_factor, bias)
        self.multi_scale_fusion_level2 = LGFF(dim*7, dim*2, ffn_expansion_factor, bias)
    
        self.decoder = Embeddings_depth_output(dim, num_blocks, num_refinement_blocks, 
                                         num_heads, bias)


    def forward(self, x, depth):
        
        hx, res1, res2 = self.encoder(x)
        
        res2_1 = F.interpolate(res2, scale_factor=2)
        res1_2 = F.interpolate(res1, scale_factor=0.5)
        hx_2   = F.interpolate(hx, scale_factor=2)
        hx_1   = F.interpolate(hx_2, scale_factor=2)
        
        res1 = self.multi_scale_fusion_level1(torch.cat((res1, res2_1, hx_1), dim=1))
        res2 = self.multi_scale_fusion_level2(torch.cat((res1_2, res2, hx_2), dim=1))

        hx = self.decoder(hx, depth, res1, res2)

        return hx + x
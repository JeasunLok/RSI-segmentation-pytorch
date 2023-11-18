import torch
from mmcv.cnn import ConvModule, xavier_init
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from model.mscan import MSCAN
from model.ham_head import LightHamHead

class SegNext_SegModel(nn.Module):
    def __init__(self,
                input_channels=3,
                out_channels=11,
                embed_dims=[64, 128, 320, 512],
                mlp_ratios=[4, 4, 4, 4],
                drop_rate=0.,
                drop_path_rate=0.,
                depths=[3, 5, 27, 3],
                num_stages=4,
                attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                act_cfg=dict(type='GELU'),
                #  norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_cfg=dict(type='BN', requires_grad=True),
                pretrained=None,
                init_cfg=None,
                least_downsample_factor=8, 
                in_channels=[128, 320, 512], 
                in_index=[1, 2, 3], 
                ham_channels=1024, 
                ham_kwargs=dict()):
        
        super(SegNext_SegModel, self).__init__()

        # backbone encoder
        self.encoder = MSCAN(input_channels,
                embed_dims,
                mlp_ratios,
                drop_rate,
                drop_path_rate,
                depths,
                num_stages,
                attention_kernel_sizes,
                attention_kernel_paddings,
                act_cfg,
                norm_cfg,
                pretrained,
                init_cfg)

        # decoder
        self.decoder = LightHamHead(least_downsample_factor, 
                in_channels,
                out_channels, 
                in_index, 
                ham_channels, 
                ham_kwargs)

    def forward(self, x):

        x = self.encoder(x)

        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # print(x[3].shape)

        x = self.decoder(x)

        # print(x.shape)

        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    model = SegNext_SegModel(out_channels=3).to(device)
    input = torch.randn([2,3,512,512]).to(device)
    output = model(input)
    print("="*100)
    print(output.shape)
    # print(len(output))
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)
    # print(output[3].shape)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6)) 
    print("="*100)
    

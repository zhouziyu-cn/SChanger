import torch
import torch.nn as nn
from typing import Callable, Optional
from functools import partial
from torch import Tensor
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math


class LFTM(nn.Module):
    """ Lightweight Feature Enhancement Module"""
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 dropout_rate: float):
        super(LFTM, self).__init__()
        self.has_shortcut = (in_channel == out_channel)
        expand_ratio = 6
        activation_layer = nn.SiLU  
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        expanded_c = in_channel * expand_ratio
        se_ratio =0.25
        self.expand_conv = ConvBNAct(in_channel,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=3,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_channel,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity) 
        
        self.se = SqueezeExcite(in_channel, expanded_c, se_ratio)
        self.drop_rate = dropout_rate
        if self.has_shortcut and dropout_rate > 0:
            self.dropout = DropPath(dropout_rate)

    def forward(self, input):
        t1, t2 = input
        result1 = self.expand_conv(t1)
        result1 = self.dwconv(result1)
        result1 = self.se(result1)
        result1 = self.project_conv(result1)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result1 = self.dropout(result1)
            result1 += t1

        result2 = self.expand_conv(t2)
        result2 = self.dwconv(result2)
        result2 = self.se(result2)
        result2 = self.project_conv(result2)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result2 = self.dropout(result2)
            result2 += t2
        return (result1,result2)

class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result
    
class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class SCLKA(nn.Module):
    ''' Spatial Consistency Large Kernel Attention'''
    def __init__(self, dim):
        super().__init__()

        self.diff = TFM(dim*2)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, input):
        t1, t2 = input
        t1_skip, t2_skip =t1.clone(), t2.clone()
        attn = self.diff((t1, t2))
        attn = self.conv0(attn)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return (t1_skip * attn,t2_skip*attn)
    
class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SCLKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, input):
        t1, t2 = input
        t1_skip, t2_skip =t1.clone(), t2.clone()  
        t1 = self.proj_1(t1)
        t2 = self.proj_1(t2)
        t1 = self.activation(t1)
        t2 = self.activation(t2)
        t1,t2 = self.spatial_gating_unit((t1,t2))
        t1 = self.proj_2(t1)
        t2 = self.proj_2(t2)
        t1 = t1 + t1_skip
        t2 = t2 + t2_skip
        return (t1, t2)
    
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class SCAM(nn.Module):
    """Spatial Consistency Attention Module"""
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, input):
        t1, t2 = input
        t1,t2 = self.attn((self.norm1(t1),self.norm1(t2)))
        t1_skip,t2_skip = input
        t1 = t1_skip + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * t1)
        t2 = t2_skip + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * t2)
        t1 = t1 + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(t1)))
        t2 = t2 + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(t2)))
        return (t1,t2)
    
class Head(nn.Module):
    def __init__(self,int_ch: int, out_ch: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(int_ch, out_ch, kernel_size=3,padding=1)
    def forward(self,x):
        return self.conv(x)
    
class Head2(nn.Module):
    def __init__(self,int_ch: int, out_ch: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(int_ch, out_ch, kernel_size=1)
    def forward(self,x):
        return self.conv(x)
    
class TFM(nn.Module):
    def __init__(self,int_ch: int, use_conv = True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.diff = nn.Sequential(
            nn.Conv2d(int_ch, int_ch//2,1),
            LayerNorm(normalized_shape=int_ch//2, data_format='channels_first'),
            nn.GELU()
            )
    def forward(self,input):
        x1,x2 = input
        if self.use_conv:
            return self.diff(torch.cat([x1,x2],dim=1))
        else:
            return x1-x2

            


    
class SChanger(nn.Module):
    def __init__(self,num_classes=1, input_channels=3, c_list=[8*3,8*4,8*6,8*8,8*13,8*15],dropout=0.2):
        super().__init__()
            
        self.encoder1 = nn.Sequential(
            LFTM(in_channel=c_list[0], out_channel=c_list[1],dropout_rate = dropout),
            LFTM(in_channel=c_list[1], out_channel=c_list[1],dropout_rate = dropout),
        )
        self.encoder2 = nn.Sequential(
            LFTM(in_channel=c_list[1], out_channel=c_list[2],dropout_rate = dropout),
            LFTM(in_channel=c_list[2], out_channel=c_list[2],dropout_rate = dropout),
        )
        self.encoder3 = nn.Sequential(
            LFTM(in_channel=c_list[2], out_channel=c_list[3],dropout_rate = dropout),
            LFTM(in_channel=c_list[3], out_channel=c_list[3],dropout_rate = dropout),
        )
        self.encoder4 = nn.Sequential(
            LFTM(in_channel=c_list[3], out_channel=c_list[4],dropout_rate = dropout),
            LFTM(in_channel=c_list[4], out_channel=c_list[4],dropout_rate = dropout),
        )
        self.encoder5 = nn.Sequential(
            LFTM(in_channel=c_list[4], out_channel=c_list[5],dropout_rate = dropout),
            LFTM(in_channel=c_list[5], out_channel=c_list[5],dropout_rate = dropout),
        )
        self.decoder1 = nn.Sequential(
            LFTM(in_channel=c_list[5], out_channel=c_list[5],dropout_rate = dropout),
            LFTM(in_channel=c_list[5], out_channel=c_list[4],dropout_rate = dropout),
        ) 
        self.decoder2 = nn.Sequential(
            LFTM(in_channel=c_list[4], out_channel=c_list[4],dropout_rate = dropout),
            LFTM(in_channel=c_list[4], out_channel=c_list[3],dropout_rate = dropout),
        ) 
        self.decoder3 = nn.Sequential(
            LFTM(in_channel=c_list[3], out_channel=c_list[3],dropout_rate = dropout),
            LFTM(in_channel=c_list[3], out_channel=c_list[2],dropout_rate = dropout),
        ) 
        self.decoder4 = nn.Sequential(
            LFTM(in_channel=c_list[2], out_channel=c_list[2],dropout_rate = dropout),
            LFTM(in_channel=c_list[2], out_channel=c_list[1],dropout_rate = dropout),
        ) 
        self.decoder5 = nn.Sequential(
            LFTM(in_channel=c_list[1], out_channel=c_list[1],dropout_rate = dropout),
            LFTM(in_channel=c_list[1], out_channel=c_list[0],dropout_rate = dropout),
        ) 

        self.head1 = nn.Sequential(TFM(c_list[4]*2),
                                   Head(c_list[4]))
        self.head2 = nn.Sequential(TFM(c_list[3]*2),
                                   Head(c_list[3]))
        self.head3 = nn.Sequential(TFM(c_list[2]*2),
                                   Head(c_list[2]))
        self.head4 = nn.Sequential(TFM(c_list[1]*2),
                                   Head(c_list[1]))
        self.head5 = nn.Sequential(TFM(c_list[0]*2),
                                   Head(c_list[0]))
        self.head6 = nn.Sequential(Head2(5,num_classes))

        #self.ChannelExchange = ChannelExchange(p=2)


        self.Bi1 = nn.Sequential(
            SCAM(c_list[5], mlp_ratio=4., drop=dropout,drop_path=dropout, act_layer=nn.GELU)
        ) 
        self.Bi2 = nn.Sequential(
        SCAM(c_list[4], mlp_ratio=4., drop=dropout,drop_path=dropout, act_layer=nn.GELU)
        ) 
        self.Bi3 = nn.Sequential(
        SCAM(c_list[3], mlp_ratio=4., drop=dropout,drop_path=dropout, act_layer=nn.GELU)
        ) 
        self.Bi4 = nn.Sequential(
        SCAM(c_list[2], mlp_ratio=4., drop=dropout,drop_path=dropout, act_layer=nn.GELU)
        ) 
        self.Bi5 = nn.Sequential(
        SCAM(c_list[1], mlp_ratio=4., drop=dropout,drop_path=dropout, act_layer=nn.GELU)
        )  

        self.conv_1 = ConvBNAct(input_channels,
                                c_list[0],
                                kernel_size=3,)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input):
        x1, x2 = input
        _,c,h,w = x1.shape
        encode_outputs1 = []
        encode_outputs2 = []
        outputs =  []
        x1,x2 = self.encoder1((self.conv_1(x1),self.conv_1(x2)))
        encode_outputs1.append(x1)
        encode_outputs2.append(x2)

        x1,x2 = self.encoder2((F.max_pool2d(x1,2,2),F.max_pool2d(x2,2,2)))
        encode_outputs1.append(x1)
        encode_outputs2.append(x2)

        x1,x2 = self.encoder3((F.max_pool2d(x1,2,2),F.max_pool2d(x2,2,2)))
        encode_outputs1.append(x1)
        encode_outputs2.append(x2)

        x1,x2 = self.encoder4((F.max_pool2d(x1,2,2),F.max_pool2d(x2,2,2)))
        encode_outputs1.append(x1)
        encode_outputs2.append(x2)

        x1,x2 = self.encoder5((F.max_pool2d(x1,2,2),F.max_pool2d(x2,2,2)))
        

        x1,x2 =  self.Bi1((x1,x2))
        x1,x2 = self.decoder1((x1,x2))
        mask1 = self.head1((x1,x2))
        outputs.append(F.interpolate(mask1, size=[h, w], mode='bilinear', align_corners=True))

        x1_skip = encode_outputs1.pop()
        x2_skip = encode_outputs2.pop()
        x1_skip,x2_skip =  self.Bi2((x1_skip,x2_skip))
        x1 = F.interpolate(x1,scale_factor=(2,2),mode ='bilinear',align_corners=True) +x1_skip
        x2 = F.interpolate(x2,scale_factor=(2,2),mode ='bilinear',align_corners=True) +x2_skip
        x1,x2 = self.decoder2((x1,x2))
        mask2 = self.head2((x1,x2))
        outputs.append(F.interpolate(mask2, size=[h, w], mode='bilinear', align_corners=True))

        x1_skip = encode_outputs1.pop()
        x2_skip = encode_outputs2.pop()
        x1_skip,x2_skip =  self.Bi3((x1_skip,x2_skip))
        x1 = F.interpolate(x1,scale_factor=(2,2),mode ='bilinear',align_corners=True) +x1_skip
        x2 = F.interpolate(x2,scale_factor=(2,2),mode ='bilinear',align_corners=True) +x2_skip
        x1,x2 = self.decoder3((x1,x2))
        mask3 = self.head3((x1,x2))
        outputs.append(F.interpolate(mask3, size=[h, w], mode='bilinear', align_corners=True))

        x1_skip = encode_outputs1.pop()
        x2_skip = encode_outputs2.pop()
        x1_skip,x2_skip =  self.Bi4((x1_skip,x2_skip))
        x1 = F.interpolate(x1,scale_factor=(2,2),mode ='bilinear',align_corners=True) +x1_skip
        x2 = F.interpolate(x2,scale_factor=(2,2),mode ='bilinear',align_corners=True) +x2_skip
        x1,x2 = self.decoder4((x1,x2))
        mask4 = self.head4((x1,x2))
        outputs.append(F.interpolate(mask4, size=[h, w], mode='bilinear', align_corners=True))

        x1_skip = encode_outputs1.pop()
        x2_skip = encode_outputs2.pop()
        x1_skip,x2_skip =  self.Bi5((x1_skip,x2_skip))
        x1 = F.interpolate(x1,scale_factor=(2,2),mode ='bilinear',align_corners=True) +x1_skip
        x2 = F.interpolate(x2,scale_factor=(2,2),mode ='bilinear',align_corners=True) +x2_skip
        x1,x2 = self.decoder5((x1,x2))
        mask5 = self.head5((x1,x2))
        outputs.append(F.interpolate(mask5, size=[h, w], mode='bilinear', align_corners=True))
        outputs = torch.cat(outputs,dim=1)
        if self.training:
            return torch.cat([self.head6(outputs), outputs], dim=1)
        else:
            return torch.sigmoid(self.head6(outputs))

weight_urls = {
    'levir-cd': {
        'schanger-base': 'https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_base_levir.pth',
        'schanger-small': "https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_small_levir.pth",
    },
    'levir-cd+':{
        'schanger-base':'https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_base_levir_plus.pth',
        'schanger-small':'https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_small_levir_plus.pth',
    },
    's2looking':{
        'schanger-base':'https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_base_s2looking.pth',
        'schanger-small':'https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_small_s2looking.pth',
    },
    'cdd':{
        'schanger-base':'https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_base_cdd.pth',
        'schanger-small':'https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_small_cdd.pth',
    },
    'whu-cd':{
        'schanger-base':'https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_base_whu.pth',
        'schanger-small':'https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_small_whu.pth',

    },
    'sysu-cd':{
        'schanger-base':'https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_base_sysu.pth',
        'schanger-small':'https://huggingface.co/Zy-Zhou/schanger/resolve/main/schanger_small_sysu.pth',
    }
}
def get_url(dataset_name, model_name):
    if dataset_name in weight_urls:
        dataset = weight_urls[dataset_name]
        if model_name in dataset:
            return dataset[model_name]
        else:
            assert f"Model '{model_name}' not found in dataset '{dataset_name}'."
    else:
        assert f"Dataset '{dataset_name}' not found."
    
def schanger_small(dataset_name,pretrained=False):
    model = SChanger(num_classes=1, input_channels=3, c_list=[8*1,8*2,8*4,8*5,8*6,8*6],dropout=0.2)
    if pretrained:
        url = get_url(dataset_name,'schanger-small')
        state_dict = torch.hub.load_state_dict_from_url(url,progress=True)
        model.load_state_dict(state_dict['state_dict'])
    return model


def schanger_base(pretrained=False,dataset_name='LEVIR-CD'):
    model = SChanger(num_classes=1, input_channels=3, c_list=[8*3,8*4,8*6,8*8,8*13,8*15],dropout=0.2)
    if pretrained:
        url = get_url(dataset_name,'schanger-base')
        state_dict = torch.hub.load_state_dict_from_url(url,progress=True)
        model.load_state_dict(state_dict['state_dict'])
    return model


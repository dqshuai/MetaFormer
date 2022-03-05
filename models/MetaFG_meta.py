import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from .MBConv import MBConvBlock
from .MHSA import MHSABlock,Mlp
from .meta_encoder import ResNormLayer
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'MetaFG_0': _cfg(),
    'MetaFG_1': _cfg(),
    'MetaFG_2': _cfg(),
}

def make_blocks(stage_index,depths,embed_dims,img_size,dpr,extra_token_num=1,num_heads=8,mlp_ratio=4.,stage_type='conv'):
    stage_name = f'stage_{stage_index}'
    blocks = []
    for block_idx in range(depths[stage_index]):
        stride = 2 if block_idx == 0 and stage_index != 1 else 1
        in_chans = embed_dims[stage_index] if block_idx != 0 else  embed_dims[stage_index-1]
        out_chans = embed_dims[stage_index]
        image_size = img_size if block_idx == 0 or stage_index == 1 else img_size//2
        drop_path_rate = dpr[sum(depths[1:stage_index])+block_idx]
        if stage_type == 'conv':
            blocks.append(MBConvBlock(ksize=3,input_filters=in_chans,output_filters=out_chans,
                                      image_size=image_size,expand_ratio=int(mlp_ratio),stride=stride,drop_connect_rate=drop_path_rate))
        elif stage_type == 'mhsa':
            blocks.append(MHSABlock(input_dim=in_chans,output_dim=out_chans,
                                    image_size=image_size,stride=stride,num_heads=num_heads,extra_token_num=extra_token_num,
                                    mlp_ratio=mlp_ratio,drop_path=drop_path_rate))
        else:
            raise NotImplementedError("We only support conv and mhsa")
    return blocks
    

class MetaFG_Meta(nn.Module):
    def __init__(self,img_size=224,in_chans=3, num_classes=1000,
                conv_embed_dims = [64,96,192],attn_embed_dims=[384,768],
                conv_depths = [2,2,3],attn_depths = [5,2],num_heads=32,extra_token_num=3,mlp_ratio=4.,
                conv_norm_layer=nn.BatchNorm2d,attn_norm_layer=nn.LayerNorm,
                conv_act_layer=nn.ReLU,attn_act_layer=nn.GELU,
                qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.,
                add_meta=True,meta_dims=[4,3],mask_prob=1.0,mask_type='linear',
                only_last_cls=False,
                use_checkpoint=False):
        super().__init__()
        self.only_last_cls = only_last_cls
        self.img_size = img_size
        self.num_classes = num_classes
        self.add_meta = add_meta
        self.meta_dims = meta_dims
        self.cur_epoch = -1
        self.total_epoch = -1
        self.mask_prob = mask_prob
        self.mask_type = mask_type
        self.attn_embed_dims = attn_embed_dims
        self.extra_token_num = extra_token_num
        if self.add_meta:
#             assert len(meta_dims)==extra_token_num-1
            for ind,meta_dim in enumerate(meta_dims):
                meta_head_1 = nn.Sequential(
                                        nn.Linear(meta_dim, attn_embed_dims[0]),
                                        nn.ReLU(inplace=True),
                                        nn.LayerNorm(attn_embed_dims[0]),
                                        ResNormLayer(attn_embed_dims[0]),
                                        ) if meta_dim > 0 else nn.Identity()
                meta_head_2 = nn.Sequential(
                                        nn.Linear(meta_dim, attn_embed_dims[1]),
                                        nn.ReLU(inplace=True),
                                        nn.LayerNorm(attn_embed_dims[1]),
                                        ResNormLayer(attn_embed_dims[1]),
                                        ) if meta_dim > 0 else nn.Identity()  
                setattr(self, f"meta_{ind+1}_head_1", meta_head_1)
                setattr(self, f"meta_{ind+1}_head_2", meta_head_2)
        
        
        stem_chs = (3 * (conv_embed_dims[0] // 4), conv_embed_dims[0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(conv_depths[1:]+attn_depths))]
        #stage_0
        self.stage_0 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                conv_norm_layer(stem_chs[0]),
                conv_act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                conv_norm_layer(stem_chs[1]),
                conv_act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], conv_embed_dims[0], 3, stride=1, padding=1, bias=False)])
        self.bn1 = conv_norm_layer(conv_embed_dims[0])
        self.act1 = conv_act_layer(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #stage_1
        self.stage_1 = nn.ModuleList(make_blocks(1,conv_depths+attn_depths,conv_embed_dims+attn_embed_dims,img_size//4,
                                      dpr=dpr,num_heads=num_heads,extra_token_num=extra_token_num,mlp_ratio=mlp_ratio,stage_type='conv'))
        #stage_2
        self.stage_2 = nn.ModuleList(make_blocks(2,conv_depths+attn_depths,conv_embed_dims+attn_embed_dims,img_size//4,
                                      dpr=dpr,num_heads=num_heads,extra_token_num=extra_token_num,mlp_ratio=mlp_ratio,stage_type='conv'))
        
        #stage_3
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, attn_embed_dims[0]))
        self.stage_3 = nn.ModuleList(make_blocks(3,conv_depths+attn_depths,conv_embed_dims+attn_embed_dims,img_size//8,
                                      dpr=dpr,num_heads=num_heads,extra_token_num=extra_token_num,mlp_ratio=mlp_ratio,stage_type='mhsa'))
        #stage_4
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, attn_embed_dims[1]))
        self.stage_4 = nn.ModuleList(make_blocks(4,conv_depths+attn_depths,conv_embed_dims+attn_embed_dims,img_size//16,
                                      dpr=dpr,num_heads=num_heads,extra_token_num=extra_token_num,mlp_ratio=mlp_ratio,stage_type='mhsa'))
        self.norm_2 = attn_norm_layer(attn_embed_dims[1])
        
        #Aggregate
        if not self.only_last_cls:
            self.cl_1_fc = nn.Sequential(*[Mlp(in_features=attn_embed_dims[0], out_features=attn_embed_dims[1]),
                                         attn_norm_layer(attn_embed_dims[1])])
            self.aggregate = torch.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
            self.norm = attn_norm_layer(attn_embed_dims[1])
            self.norm_1 = attn_norm_layer(attn_embed_dims[0])
        # Classifier head
        self.head = nn.Linear(attn_embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        trunc_normal_(self.cls_token_1, std=.02)
        trunc_normal_(self.cls_token_2, std=.02)
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
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token_1','cls_token_2'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self,x,meta=None):
        B = x.shape[0]
        extra_tokens_1 = [self.cls_token_1]
        extra_tokens_2 = [self.cls_token_2]
        if self.add_meta:
            assert meta != None,'meta is None'
            if len(self.meta_dims)>1:
                metas = torch.split(meta,self.meta_dims,dim=1)
            else:
                metas = (meta,)
            for ind,cur_meta in enumerate(metas):
                meta_head_1 = getattr(self,f"meta_{ind+1}_head_1")
                meta_head_2 = getattr(self,f"meta_{ind+1}_head_2")
                meta_1 = meta_head_1(cur_meta)
                meta_1 = meta_1.reshape(B, -1, self.attn_embed_dims[0])
                meta_2 = meta_head_2(cur_meta)
                meta_2 = meta_2.reshape(B, -1, self.attn_embed_dims[1])
                extra_tokens_1.append(meta_1)
                extra_tokens_2.append(meta_2)
            
        x = self.stage_0(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        for blk in self.stage_1:
            x = blk(x)
        for blk in self.stage_2:
            x = blk(x)
        H0,W0 = self.img_size//8,self.img_size//8
        for ind,blk in enumerate(self.stage_3):
            if ind==0:
                x = blk(x,H0,W0,extra_tokens_1)
            else:
                x = blk(x,H0,W0)
        if not self.only_last_cls:
            cls_1 = x[:, :1, :]
            cls_1 = self.norm_1(cls_1)
            cls_1 = self.cl_1_fc(cls_1)
        
        x = x[:, self.extra_token_num:, :]
        H1,W1 = self.img_size//16,self.img_size//16
        x = x.reshape(B,H1,W1,-1).permute(0, 3, 1, 2).contiguous()
        for ind,blk in enumerate(self.stage_4):
            if ind==0:
                x = blk(x,H1,W1,extra_tokens_2)
            else:
                x = blk(x,H1,W1)
        cls_2 = x[:, :1, :]
        cls_2 = self.norm_2(cls_2)
        if not self.only_last_cls:
            cls = torch.cat((cls_1,cls_2), dim=1)#B,2,C
            cls = self.aggregate(cls).squeeze(dim=1)#B,C
            cls = self.norm(cls)
        else:
            cls = cls_2.squeeze(dim=1)
        return cls
    def forward(self, x,meta=None):
        if meta is not None:
            if self.mask_type=='linear':
                cur_mask_prob = self.mask_prob - self.cur_epoch/self.total_epoch
            else:
                cur_mask_prob = self.mask_prob
            if cur_mask_prob != 0 and self.training:
                mask = torch.ones_like(meta)
                mask_index = torch.randperm(meta.size(0))[:int(meta.size(0)*cur_mask_prob)]
                mask[mask_index] = 0
                meta = mask * meta
        x = self.forward_features(x,meta)
        x = self.head(x)
        return x 

@register_model
def MetaFG_meta_0(pretrained=False, **kwargs):
    model = MetaFG_Meta(conv_embed_dims = [64,96,192],attn_embed_dims=[384,768],
                 conv_depths = [2,2,3],attn_depths = [5,2],num_heads=8,mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['MetaFG_0']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
@register_model
def MetaFG_meta_1(pretrained=False, **kwargs):
    model = MetaFG_Meta(conv_embed_dims = [64,96,192],attn_embed_dims=[384,768],
                 conv_depths = [2,2,6],attn_depths = [14,2],num_heads=8,mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['MetaFG_1']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
@register_model
def MetaFG_meta_2(pretrained=False, **kwargs):
    model = MetaFG_Meta(conv_embed_dims = [128,128,256],attn_embed_dims=[512,1024],
                 conv_depths = [2,2,6],attn_depths = [14,2],num_heads=8,mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['MetaFG_2']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
if __name__ == "__main__":
    x = torch.randn([2, 3, 224, 224])
    meta = torch.randn([2,7])
    model = MetaFG_meta()
    import ipdb;ipdb.set_trace()
    output = model(x,meta)
    print(output.shape)
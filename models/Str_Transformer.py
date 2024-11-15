from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import numpy as np
import torch.nn.functional as F
from torch_cluster import fps
import math
from timm.models.layers import LayerNorm2d
import gcn3d 
import time



class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=-2, keepdim=False)
        return x.pow(1. / self.p).squeeze(-1)

class GCN3D(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int):
        super().__init__()

        # GCN
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 32, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(32, 64, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(64, 128, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)

        self.ln0 = nn.LayerNorm([32], eps=1e-6)
        self.ln1 = nn.LayerNorm([64], eps=1e-6)
        self.ln2 = nn.LayerNorm([128], eps=1e-6)
        self.ln3 = nn.LayerNorm([256], eps=1e-6)
        self.ln4 = nn.LayerNorm([1024], eps=1e-6)
        self.bn = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn1 = nn.BatchNorm2d(1024)

        # BoQ
        self.token = torch.nn.Parameter(torch.randn(1, 64, 1024, 1))
        self.pool_3 = gcn3d.Pool_layer(pooling_rate= 16, neighbor_num= 16)
        self.dwconv_2 = nn.Conv2d(64,64,(1,1), stride = 1, groups=64)
        self.conv_5 = nn.Conv2d(64,256,(1,1))
        self.conv_6 = nn.Conv2d(256,64,(1,1))

        # SIM
        self.gcn1 = gcn3d.Conv_layer(64, 256, support_num= support_num)
        self.gcn2 = gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.ffn_1 = nn.Conv2d(64,256,(1,1))
        self.ffn_2 = nn.Conv2d(256,256,(1,1))
        self.cross_attn_1 = torch.nn.MultiheadAttention(64, num_heads=4, batch_first=True) # [batch_size, sequence_length, features]

        # CIM
        self.cross_attn_2 = torch.nn.MultiheadAttention(256, num_heads=4, batch_first=True) # [batch_size, sequence_length, features]
     
        # FFN        
        self.conv_7 = gcn3d.Conv_layer(256, 1024, support_num= support_num)
        self.conv_8 = nn.Conv2d(1024, 256, (1,1))

    def forward(self,  vertices: "(bs, 1, vertice_num, 3)"):

        # Backbone
        bs, _, vertice_num, _ = vertices.size()
        vertices_0 = vertices.view(bs, vertice_num, 3) 
        
        # gcn layer1
        neighbor_index_0 = gcn3d.get_neighbor_index(vertices_0, self.neighbor_num)
        fm_0 = F.relu(self.ln0(self.conv_0(neighbor_index_0, vertices_0)))
        fm_1 = F.relu(self.ln1(self.conv_1(neighbor_index_0, vertices_0, fm_0)))

        # pooling1
        vertices_1, fm_1 = self.pool_1(vertices_0, fm_1)
        neighbor_index_1 = gcn3d.get_neighbor_index(vertices_1, self.neighbor_num)
        
        # X
        fm_x = fm_1

        # gcn layer2
        fm_2 = F.relu(self.ln2(self.conv_2(neighbor_index_1, vertices_1, fm_1)))
        fm_3 = F.relu(self.ln3(self.conv_3(neighbor_index_1, vertices_1, fm_2)))
     
        # pooling2
        _, fm_3 = self.pool_2(vertices_1, fm_3) # [bs, vertice_num, out_channel]
    
        # query
        B = fm_3.size(0)
        token = self.token.repeat(B, 1, 1, 1)
        token = token.squeeze(-1).permute(0,2,1)

        # CIM
        query = self.cross_attn_1(token,fm_x,fm_x)[0] + fm_x

        #FFN_Conv
        query = query.permute(0,2,1).unsqueeze(-1)
        query = self.ffn_2(F.relu(self.bn2(self.ffn_1(query))))
        query = query.squeeze(-1).permute(0,2,1)
        
        # CIM
        cross_attn = self.cross_attn_2(query, fm_3, fm_3)[0] + query # [bs, vertice_num, out_channel]

        # FFN      
        fm_final = F.relu(self.ln4(self.conv_7(neighbor_index_1, vertices_1, cross_attn)))

        fm_final = fm_final.permute(0,2,1).unsqueeze(-1)

        fm_final = self.conv_8(fm_final)

        return fm_final

class Str_Transformer(nn.Module):
    def __init__(self):
        super(Str_Transformer, self).__init__()
        self.point_net =GCN3D(support_num= 1, neighbor_num= 20)
        self.gem = GeM()

    def forward(self, x):
        x = self.point_net(x)
        x = self.gem(x)
        return x

# if __name__ == '__main__':
#     num_points = 4096
#     sim_data = Variable(torch.rand(44, 1, num_points, 3))
#     sim_data = sim_data

#     pnv = models.Str_Transformer(global_feat=True, feature_transform=True, max_pool=False,
#                                     output_dim=256, num_points=num_points)
#     pnv.train()
#     out3 = pnv(sim_data)
#     print('pnv', out3.size())

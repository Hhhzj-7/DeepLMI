import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, SGConv, FiLMConv, GATv2Conv, GINConv
from torch_geometric.nn import global_mean_pool

class GCN_miRNA(nn.Module):
    def __init__(self, hidden_dim):
        super(GCN_miRNA, self).__init__()
        self.conv1 =  GCNConv(640, hidden_dim)
        self.conv3 =  GCNConv(hidden_dim , hidden_dim)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.2)
    def forward(self, batch):
        edge_index = batch.edge_index
        x_emb = batch.emb
        x1 = self.dropout(self.relu(self.conv1((x_emb), edge_index)))
        x1 = self.dropout((self.conv3(x1, edge_index)))
        output = global_mean_pool(x1, batch.batch)
        return output


class SA_lncRNA(nn.Module):
    def __init__(self, hidden_dim):
        super(SA_lncRNA, self).__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim*2, bias=False)
        self.l2 = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)
        self.model_kf = SelfAttention(hidden_dim)

    def forward(self, x):
        x1, x2, x3 = x
        x = self.model_kf(torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)), dim=1))
        x = torch.mean(x,1).squeeze(1)
        output = self.l2(self.dropout(self.relu(self.l1(x))))

        return output


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 100)
        self.conv2 = GCNConv(100, 100)

        self.relu = nn.ReLU()
        self.l1 = nn.Linear(200, 100, bias=False)
        self.l2 = nn.Linear(100, 1, bias=False)

    def encode(self, x, edge_index, type_lnc_idx, type_mi_idx):
        
        X = x

        x = self.relu(self.conv1(x, edge_index))
        x = x.clone()
        virtual_a = x[type_lnc_idx].mean(dim=0, keepdim=True)
        x[type_lnc_idx] = x[type_lnc_idx] + virtual_a

        virtual_b = x[type_mi_idx].mean(dim=0, keepdim=True)
        x = x.clone()
        x[type_mi_idx] = x[type_mi_idx] + virtual_b

        x = self.conv2(x, edge_index)
        
        return (x, X)

    def decode(self, z, edge_label_index):
        x_g, x_self = z
        start_node_features_g = x_g[edge_label_index[0]]
        end_node_features_g = x_g[edge_label_index[1]]

        start_node_features_self = x_self[edge_label_index[0]]
        end_node_features_self = x_self[edge_label_index[1]]

        edge_features = torch.cat([(start_node_features_g + start_node_features_self)/2,
                                   (end_node_features_g + end_node_features_self)/2], dim=1)
        x1 = self.relu(self.l1(edge_features))
        x = self.l2(x1).squeeze(1)
        return x, x1

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()



class SelfAttention(nn.Module):
    def __init__(self, embed_size=100, heads=1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embedding size must be divisible by heads"

        # 定义线性变换层，用于生成查询（queries）、键（keys）和值（values）
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N = x.shape[0]
        length = x.shape[1]

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, length, self.heads, self.head_dim)
        keys = keys.view(N, length, self.heads, self.head_dim)
        queries = queries.view(N, length, self.heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, length, self.heads * self.head_dim
        )

        return self.fc_out(out)

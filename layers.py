import torch 
import torch.nn as nn
import torch.nn.functional as F 

from torch_scatter import scatter 

import IPython 

class SNAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(torch.cuda.is_available()):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
        return None, grad_values, None, None, None


class SparseNeighborhoodAggregation(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SNAFunction.apply(edge, edge_w, N, E, out_features)
    

class GAT(nn.Module):
    def __init__(self, n_ent, in_dim, out_dim, dropout=0.5, margin=6.0, 
        epsilon=2.0, device="cuda", concat=True):
        super().__init__() 

        self.n_ent = n_ent 
        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.device = device 

        self.margin = margin 
        self.epsilon = epsilon 

        self.a = nn.Linear(in_dim, out_dim).to(device)
        nn.init.xavier_normal_(self.a.weight.data, gain=1.414)

        self.a_2 = nn.Linear(out_dim, 1).to(device)
        nn.init.xavier_normal_(self.a_2.weight.data, gain=1.414)

        self.sparse_neighborhood_aggregation = SparseNeighborhoodAggregation()

        self.concat = concat 

        if concat:
            self.ent_embed_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.out_dim]),
                requires_grad=False
            )

            self.ent_embed = nn.Embedding(
                n_entities, in_dim, max_norm=1, norm_type=2).to(device)

            nn.init.uniform_(self.ent_embed.weight.data, -
                             self.ent_embed_range.item(), self.ent_embed_range.item())

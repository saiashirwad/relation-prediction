import torch
import torch.nn as nn
import torch.nn.functional as F 
from .Model import Model

from torch_scatter import scatter

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


class KGLayer(nn.Module):
    def __init__(self, n_entities, n_relations, in_dim, out_dim, input_drop=0.5, 
                 margin=6.0, epsilon=2.0, device="cuda", concat=True):
        super().__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        self.margin = margin 
        self.epsilon = epsilon

        self.a = nn.Linear(3 * in_dim, out_dim).to(device)
        nn.init.xavier_normal_(self.a.weight.data, gain=1.414)

        self.a_2 = nn.Linear(out_dim, 1).to(device)
        nn.init.xavier_normal_(self.a_2.weight.data, gain=1.414)

        self.sparse_neighborhood_aggregation = SparseNeighborhoodAggregation()

        self.concat = concat 

        if concat:
            self.ent_embed_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.out_dim]), 
                requires_grad = False
            )
            
            self.rel_embed_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.out_dim]),
                requires_grad = False
            )
    
            self.ent_embed = nn.Embedding(n_entities, in_dim, max_norm=1, norm_type=2).to(device)
            self.rel_embed = nn.Embedding(n_relations, in_dim, max_norm=1, norm_type=2).to(device)
            
            nn.init.uniform_(self.ent_embed.weight.data, -self.ent_embed_range.item(), self.ent_embed_range.item())
            nn.init.uniform_(self.rel_embed.weight.data, -self.rel_embed_range.item(), self.rel_embed_range.item())

        self.input_drop = nn.Dropout(input_drop)

        self.bn0 = nn.BatchNorm1d(3 * in_dim).to(device)
        self.bn1 = nn.BatchNorm1d(out_dim).to(device)
    
    def forward(self, triplets, ent_embed=None, rel_embed=None):
        N = self.n_entities
    
        if self.concat:
            h = torch.cat((
                self.ent_embed(triplets[:, 0]),
                self.rel_embed(triplets[:, 1]),
                self.ent_embed(triplets[:, 2])
            ), dim=1)
        else:
            h = torch.cat((
                ent_embed[triplets[:, 0]],
                rel_embed[triplets[:, 1]],
                ent_embed[triplets[:, 2]]
            ), dim=1)

        h = self.input_drop(self.bn0(h))
        c = self.bn1(self.a(h))
        b = -F.leaky_relu(self.a_2(c))
        e_b = torch.exp(b) 

        temp = triplets.t()
        edges = torch.stack([temp[0], temp[2]])

        ebs = self.sparse_neighborhood_aggregation(edges, e_b, N, e_b.shape[0], 1)
        temp1 = e_b * c

        hs = self.sparse_neighborhood_aggregation(edges, temp1,  N, e_b.shape[0], self.out_dim)

        ebs[ebs == 0] = 1e-12
        h_ent = hs / ebs 

        index = triplets[:, 1]
        h_rel  = scatter(temp1, index=index, dim=0, reduce="mean") 

        return h_ent, h_rel



class RotAtte(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, margin=6.0, epsilon=2.0, n_heads=5):
        super(RotAtte, self).__init__(ent_tot, rel_tot)

        self.margin = margin
        self.epsilon = epsilon

        self.att = nn.ModuleList([
            KGAttLayer(ent_tot, rel_tot, dim, margin, epsilon)
            for _ in range(n_heads)
        ])

    def forward(self):
        pass

    def predict(self):
        pass


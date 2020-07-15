import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F

import openke.module.model as model
Model = model.Model


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
                n_ent, in_dim, max_norm=1, norm_type=2).to(device)

            nn.init.uniform_(self.ent_embed.weight.data, -
                             self.ent_embed_range.item(), self.ent_embed_range.item())

        self.dropout = nn.Dropout(dropout)

    def forward(self, triplets, ent_embed=None):
        N = self.n_ent

        if self.concat:
            h = self.ent_embed(triplets[:, 0])
        else:
            h = ent_embed[triplets[:, 0]]

        h = self.dropout(h)
        c = self.a(h)
        b = -F.leaky_relu(self.a_2(c))
        e_b = self.dropout(torch.exp(b))

        temp = triplets.t()
        edges = torch.stack([temp[0], temp[2]])

        ebs = self.sparse_neighborhood_aggregation(
            edges, e_b, N, e_b.shape[0], 1)
        temp1 = e_b * c

        hs = self.sparse_neighborhood_aggregation(
            edges, temp1, N, e_b.shape[0], self.out_dim)

        ebs[ebs == 0] = 1e-12
        h_ent = hs / ebs

        if self.concat:
            return F.relu(h_ent)
        return h_ent


class RotatE(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, margin=6.0, epsilon=2.0):
        super(RotatE, self).__init__(ent_tot, rel_tot)

        self.margin = margin
        self.epsilon = epsilon

        self.dim_e = dim
        self.dim_r = dim // 2

        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )

        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

    def forward(self, h, t, r, mode):
        pi = self.pi_const

        if type(h) == tuple:
            """
            type_ == split
            """
            re_head, im_head = h
            re_tail, im_tail = t
        else:
            re_head, im_head = torch.chunk(h, 2, dim=-1)
            re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1,
                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1,
                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1,
                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1,
                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        score = self.margin - score.permute(1, 0).flatten()

        return score

from layers import GAT, RotatE
import torch
import torch.nn as nn
import torch.nn.functional as F

import openke.module.model as model
Model = model.Model


class RotAtte(Model):
    def __init__(self, n_ent, n_rel,
                 in_dim, out_dim,
                 facts, n_heads=1, n_layers=1,
                 negative_rate=10, margin=6.0, epsilon=2.0,
                 batch_size=None, device="cuda", multiplier=-1,
                 type_="split", score="rotate"):

        super(RotAtte, self).__init__(n_ent, n_rel)

        self.n_ent = n_ent
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.facts = facts
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.negative_rate = negative_rate
        self.margin = margin
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = device
        self.multiplier = multiplier
        self.type_ = type_

        if type_ == "split":
            self.a_re = nn.ModuleList([
                GAT(n_ent, in_dim, out_dim, 0.5, margin,
                    epsilon, device, concat=True)
                for _ in range(n_heads)
            ])
            self.a_im = nn.ModuleList([
                GAT(n_ent, in_dim, out_dim, 0.5, margin,
                    epsilon, device, concat=True)
                for _ in range(n_heads)
            ])
            if self.n_heads > 1:
                self.ent_transform_re = nn.Linear(n_heads * out_dim, out_dim)
                self.ent_transform_im = nn.Linear(n_heads * out_dim, out_dim)

        elif type_ == "single":
            self.a = nn.ModuleList([
                GAT(n_ent, in_dim, out_dim, 0.5, margin,
                    epsilon, device, concat=True)
                for _ in range(n_heads)
            ])
            if self.n_heads > 1:
                self.ent_transform = nn.Linear(n_heads * out_dim, out_dim)

        self.rel_embed = nn.Embedding(n_rel, out_dim)
        self.score_fn = RotatE(n_ent, n_rel, out_dim, margin, epsilon)
        if self.score == "rotate":
            self.score_fn = RotatE(n_ent, n_rel, out_dim, margin, epsilon)
        elif self.score == "complex":
            pass
        elif self.score == "transe":
            pass

    def attention(self):
        if self.type_ == "split":
            ent_embed_re = torch.cat([a(self.facts) for a in self.a_re], dim=1)
            ent_embed_im = torch.cat([a(self.facts) for a in self.a_im], dim=1)

            if self.n_heads > 1:
                ent_embed_re = self.ent_transform_re(ent_embed_re)
                ent_embed_im = self.ent_transform_im(ent_embed_im)

            return ent_embed_re, ent_embed_im

        elif self.type_ == "single":
            ent_embed = torch.cat([a(self.facts) for a in self.a], dim=1)

            if self.n_heads > 1:
                ent_embed = self.ent_transform(ent_embed)

    def score(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        if self.type_ == "single":
            h = self.ent_embed[batch_h]
            r = self.ent_embed[batch_r]
            t = self.ent_embed[batch_t]
            return self.score_fn(h, t, r, mode)
        elif self.type_ == "split":
            h = (self.ent_embed_re[batch_h], self.ent_embed_im[batch_h])
            t = (self.ent_embed_re[batch_t], self.ent_embed_im[batch_t])
            r = self.rel_embed(batch_r)
            return self.score_fn(h, t, r, mode)

    def forward(self, data):
        if self.type_ == "single":
            ent_embed = self.attention()
            mask = torch.zeros(self.n_ent).to(self.device)
            mask_indices = torch.cat(
                (data['batch_h'], data['batch_t'])).unique()
            mask[mask_indices] = 1.0

            self.ent_embed = mask.unsqueeze(
                -1).expand_as(ent_embed) * ent_embed
        elif self.type_ == "split":
            ent_embed_re, ent_embed_im = self.attention()
            mask = torch.zeros(self.n_ent).to(self.device)
            mask_indices = torch.cat(
                (data['batch_h'], data['batch_t'])).unique()
            mask[mask_indices] = 1.0
            """
            TODO:
            Not sure if this is the cleanest/most efficient way to use
            this but I need these embeddings to be member variables
            during eval so I might as well do this here

            Does mean that self.score(data) has more magic than I am
            comfortable with, however
            """
            self.ent_embed_re = mask.unsqueeze(
                -1).expand_as(ent_embed_re) * ent_embed_re
            self.ent_embed_im = mask.unsqueeze(
                -1).expand_as(ent_embed_im) * ent_embed_im

        score = self.score(data)

        return score

    def predict(self, data):
        score = - self.score_fn(data) * self.multiplier

        return score.cpu().data.numpy()

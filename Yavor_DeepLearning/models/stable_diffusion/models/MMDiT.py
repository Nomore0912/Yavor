#!/usr/bin/env python
# -*- coding:utf-8 -*-
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from torch import nn
import torch


class SD3AdaLNZero(nn.Module):
    def __init__(
            self,
            embedding_dim,
            norm_type="layer_norm",
            bias=True,
    ):
        super(SD3AdaLNZero, self).__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 9 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.chunk(
            9, dim=1
        )
        norm_hidden_states = self.norm(hidden_states)
        # scale_msa, shift_msa = a_c, b_c
        hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
        # ???
        norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2


class AdaLNZero(nn.Module):
    def __init__(self, embedding_dim, num_embeddings=None, bias=True):
        super(AdaLNZero, self).__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
            self,
            x,
            timestep=None,
            class_labels=None,
            hidden_dtype=None,
            emb=None
    ):
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        # scale_msa, shift_msa = a_x, b_x
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class MMDiT(nn.Module):
    def __init__(
            self,
            dim,
            num_attention_heads: int,
            attention_head_dim: int,
            context_pre_only: bool = False,
    ):
        super(MMDiT, self).__init__()
        self.norm1 = SD3AdaLNZero(dim)
        self.norm1_context = AdaLNZero(dim)

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            qk_norm="rms_norm",
            eps=1e-6,
        )
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            qk_norm="rms_norm",
            eps=1e-6,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Linear(4 * dim, dim)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = nn.Linear(dim, dim)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            y,
    ):
        # x
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
            hidden_states, emb=y
        )
        # c
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=y
        )
        # Attention. latent context cross-attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
        )

        # gate_msa = r_x
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        # latent self-attention
        attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
        attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
        hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        # Mod scale_mlp, shift_mlp = x
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        # gate_mlp = Zeta_c, Zeta_x
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # c
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        # c_scale_mlp, c_shift_mlp = c
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states




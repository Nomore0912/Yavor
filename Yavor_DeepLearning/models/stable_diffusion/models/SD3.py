#!/usr/bin/env python
# -*- coding:utf-8 -*-
from torch import nn
import numpy as np
from MMDiT import MMDiT
import torch
import math


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


def get_timestep_embedding(
        time_steps,
        embedding_dim,
        scale: float = 1,
        max_period: int = 10000,
):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=time_steps.device
    )
    exponent = exponent / half_dim
    emb = torch.exp(exponent)
    emb = time_steps[:, None].float() * emb[None, :]
    # scale embeddings
    emb = scale * emb
    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def get_1d_sin_cos_pos_embed(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sin_cos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    grid_size = (grid_size, grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    emb_h = get_1d_sin_cos_pos_embed(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sin_cos_pos_embed(embed_dim // 2, grid[1])  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


class TimeSteps(nn.Module):
    def __init__(self, num_channels: int, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.scale = scale

    def forward(self, time_steps):
        t_emb = get_timestep_embedding(
            time_steps,
            self.num_channels,
            scale=self.scale,
        )
        return t_emb


class PatchEmbed(nn.Module):
    def __init__(
            self,
            height=224,
            width=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            pos_embed_max_size=1,
            bias=True,
    ):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size

        # sin cos embedding.....远古 position embedding
        self.pos_embed = get_2d_sin_cos_pos_embed(
            embed_dim, pos_embed_max_size, base_size=self.base_size, interpolation_scale=1
        )
        # Patching
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )

    def cropped_pos_embed(self, height, width):
        height = height // self.patch_size
        width = width // self.patch_size
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top: top + height, left: left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def forward(self, latent):
        height, width = latent.shape[-2:]
        # Patching
        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)
        pos_embed = self.cropped_pos_embed(height, width)
        return (latent + pos_embed).to(latent.dtype)


class CombinedTimeStepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = TimeSteps(num_channels=256)
        self.linear_1 = nn.Linear(256, embedding_dim, True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(256, 256, True)
        self.linear_3 = nn.Linear(in_features=pooled_projection_dim, out_features=embedding_dim, bias=True)
        self.act_1 = nn.SiLU()
        self.linear_4 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)

    def forward(self, time_step, pooled_projection):
        time_steps_proj = self.time_proj(time_step)
        time_steps_emb = self.linear_1(time_steps_proj)
        time_steps_emb = self.act(time_steps_emb)
        time_steps_emb = self.linear_2(time_steps_emb)

        pooled_projections = self.linear_3(pooled_projection)
        pooled_projections = self.act_1(pooled_projections)
        pooled_projections = self.linear_4(pooled_projections)

        conditioning = time_steps_emb + pooled_projections
        return conditioning


class SD3Transformer(nn.Module):
    def __init__(
            self,
            sample_size: int = 128,
            patch_size: int = 2,
            in_channels: int = 16,
            num_layers: int = 18,
            attention_head_dim: int = 64,
            num_attention_heads: int = 18,
            pos_embed_max_size: int = 96,
            dual_attention_layers: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    ):
        super(SD3Transformer, self).__init__()
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimeStepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                MMDiT(
                    dim=in_channels,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                )
                for i in dual_attention_layers
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            time_step,
    ):
        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)
        y = self.time_text_embed(time_step, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, y=y
            )
        hidden_states = self.norm_out(hidden_states, y)
        hidden_states = self.proj_out(hidden_states)

        # un patchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size
        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )
        return output

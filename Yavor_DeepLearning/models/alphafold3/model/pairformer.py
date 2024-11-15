#!/usr/bin/env python
# -*- coding:utf-8 -*-
from torch import nn
import torch


class TriangleMultiplication(nn.Module):
    def __init__(self, config):
        super(TriangleMultiplication, self).__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.linear3 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear4 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.equation = {
            'ikc,jkc->ijc': 'cik,cjk->cij',
            'kjc,kic->ijc': 'ckj,cki->cij',
        }

    def forward(self, inputs, equation):
        inputs = self.norm1(inputs)
        projection = self.act1(self.linear1(inputs)) * self.linear2(inputs)
        a, b = torch.chunk(projection, 2, dim=1)
        z = torch.einsum(self.equation[equation], a, b)
        gate = self.act2(self.linear3(inputs))
        z_out = gate * self.linear4(self.norm2(z))
        return z_out


class Transition(nn.Module):
    def __init__(self, config):
        super(Transition, self).__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.act1 = nn.SiLU()

    def forward(self, inputs):
        inputs = self.norm1(inputs)
        projection = self.relu(self.linear1(inputs))
        a, b = torch.chunk(projection, 2, dim=1)
        c = self.act1(a) * b
        inputs = self.linear2(c)
        return inputs



# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: assiene
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEncoder(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, sequence_length, device):
        super(InputEncoder, self).__init__()
        self.input_embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.max_sequence_length = sequence_length
        self.f = torch.empty(self.max_sequence_length, embedding_dim, device=device)
        nn.init.xavier_normal_(self.f)

    def forward(self, input_sequence):
        e = self.input_embedding(input_sequence)  # batch x num_sentences x num_words x embedding_dim
        s = self.f[:e.shape[2], :] * e
        s = s.sum(dim=2)  # batch x num_sentences x embedding_dim

        return s


class DynamicMemory(nn.Module):

    def __init__(self, hidden_number=20, hidden_size=100, device='cpu'):
        super(DynamicMemory, self).__init__()

        self.h = torch.empty(1, hidden_number, hidden_size, device=device)
        self.w = torch.empty(1, hidden_number, hidden_size, device=device)

        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

        self.non_linearity = nn.PReLU()  # Was PReLU, without inplace

        nn.init.xavier_normal_(self.h)
        nn.init.xavier_normal_(self.w)

    def zero_state(self, hidden_number=20, hidden_size=100, device='cpu'):
        self.h = torch.empty(1, hidden_number, hidden_size, device=device)
        nn.init.xavier_normal_(self.h)

    def forward(self, s_t):
        # s_t : batch x embedding_dim
        s_t = s_t.unsqueeze(1)
        #        print("s_t : ", s_t.shape, "\n h : ", self.h.shape)
        g = torch.sigmoid((self.h + self.w) @ s_t.transpose(1, 2))  # batch x num_memory_blocks x 1
        U = self.U(self.h)
        V = self.V(self.w)
        W = self.W(s_t)
        h_candidate = self.non_linearity(U + V + W)  # batch x num_memory_blocks x embedding_dim
        self.h = self.h + g * h_candidate  # batch x num_memory_blocks x embedding_dim
        self.h = self.h / self.h.norm(p=2, dim=2, keepdim=True)

        return self.h


class OutputModule(nn.Module):

    def __init__(self, embedding_dim, output_dim):
        super(OutputModule, self).__init__()
        self.H = nn.Linear(embedding_dim, embedding_dim)
        self.R = nn.Linear(embedding_dim, output_dim)

        self.non_linearity = nn.PReLU()

    def forward(self, q, h):
        # q : batch x embedding_dim x 1
        # h : batch x num_memory_blocks x embedding_dim

        p = F.softmax(h.bmm(q), dim=1)  # batch x num_memory_block x 1
        u = (p * h).sum(1, keepdim=True).transpose(1, 2)  # batch x embedding_dim x 1
        y = q.squeeze(2) + self.H(u.squeeze(2))  # batch x embedding_dim x 1

        return y


class RecurrentEntityNetwork(nn.Module):

    def __init__(self, vocabulary_size=177, embedding_dim=100, sequence_length=7, num_memory_blocks=20, device='cpu'):
        super(RecurrentEntityNetwork, self).__init__()
        self.device = device
        self.input_encoder = InputEncoder(vocabulary_size, embedding_dim, sequence_length, device)
        self.dynamic_memory = DynamicMemory(num_memory_blocks, embedding_dim, device=device)
        self.output_module = OutputModule(embedding_dim, vocabulary_size)

    def forward(self, text, q):
        s = self.input_encoder(text)
        q = self.input_encoder(q.unsqueeze(1))

        self.dynamic_memory.zero_state(device=self.device)
        for t in range(s.shape[1]):
            h = self.dynamic_memory(s[:, t, :])

        y = self.output_module(q.transpose(1, 2), h)

        return y

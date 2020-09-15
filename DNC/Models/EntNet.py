# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: assiene, jonathan külz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEncoder(nn.Module):

    def __init__(self, embedding_dim, sequence_length, device):
        super(InputEncoder, self).__init__()
        self.max_sequence_length = sequence_length
        self.f = torch.empty(self.max_sequence_length, embedding_dim, device=device)
        nn.init.xavier_normal_(self.f)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.f)

    def forward(self, input_sequence):
        s = self.f[:input_sequence.shape[1], :] * input_sequence  # batch x num_words x embedding_dim
        s = s.sum(dim=1)  # batch x embedding_dim

        return s


class DynamicMemory(nn.Module):

    def __init__(self, hidden_number=20, hidden_size=100, device='cpu'):
        super(DynamicMemory, self).__init__()

        self.h = torch.empty(1, hidden_number, hidden_size, device=device)
        self.w = torch.empty(1, hidden_number, hidden_size, device=device)

        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

        self.non_linearity = nn.PReLU()
        self.hidden_number = hidden_number
        self.hidden_size = hidden_size
        self.device = device

        nn.init.xavier_normal_(self.h)
        nn.init.xavier_normal_(self.w)

    def zero_state(self):
        self.h = torch.empty(1, self.hidden_number, self.hidden_size, device=self.device)
        nn.init.xavier_normal_(self.h)
        return self.h

    def reset_parameters(self):
        self.U.reset_parameters()
        self.V.reset_parameters()
        self.W.reset_parameters()
        nn.init.xavier_normal_(self.h)
        nn.init.xavier_normal_(self.w)

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

    def reset_parameters(self):
        self.H.reset_parameters()
        self.R.reset_parameters()

    def forward(self, q, h):
        # q : batch x embedding_dim x 1
        # h : batch x num_memory_blocks x embedding_dim

        p = F.softmax(h.bmm(q), dim=1)  # batch x num_memory_block x 1
        u = (p * h).sum(1, keepdim=True).transpose(1, 2)  # batch x embedding_dim x 1
        y = q.squeeze(2) + self.H(u.squeeze(2))  # batch x embedding_dim x 1

        return self.R(self.non_linearity(y))


class RecurrentEntityNetwork(nn.Module):

    def __init__(self, vocabulary_size=177, embedding_dim=100, num_memory_blocks=20, sentence_lenght=10, task='babi', device='cpu'):
        super(RecurrentEntityNetwork, self).__init__()
        self.device = device
        self.out_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.sentence_length = sentence_lenght
        self.input_encoder = InputEncoder(embedding_dim, sentence_lenght, device)
        self.dynamic_memory = DynamicMemory(num_memory_blocks, embedding_dim, device=device)
        self.output_module = OutputModule(embedding_dim, vocabulary_size)
        self.task = task

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.dynamic_memory.reset_parameters()
        self.output_module.reset_parameters()

    def requires_sentenizing(self):
        if self.task in 'babi':
            return True
        else:
            return False

    def forward(self, in_data, debug=None, embed=None, full=True):

        assert full or not self.requires_sentenizing(), "Unknown case, how to handle sentenized but not full output?"
        seq_len = in_data.shape[1] if full else 1
        out = torch.zeros([in_data.shape[0], seq_len, self.out_size], dtype=torch.float, device=self.device)
        if self.requires_sentenizing():
            for b, batch in enumerate(in_data):
                sentences = []  # (start_idx, stop_idx, type)
                start = 0
                for idx, word in enumerate(batch):
                    if word == 1:  # question mark
                        sentences.append((start, idx+1, 'question'))
                        start = idx+1
                    elif word == 3:  # dot
                        sentences.append((start, idx+1, 'info'))
                        start = idx+1
                    elif word == 0:  # think step
                        start += 1  # EntNet does not require think steps

                h = self.dynamic_memory.zero_state()
                for sentence in sentences:
                    content = batch[sentence[0]:sentence[1]]
                    if embed is not None:
                        content = embed(content)
                    content = self.input_encoder(torch.unsqueeze(content, 0))
                    if sentence[2] == 'info':
                        h = self.dynamic_memory(content)
                    else:
                        idx = sentence[1]
                        out[b, idx, :] = self.output_module(torch.unsqueeze(content, 2), h)
        else:
            h = self.dynamic_memory.zero_state()
            rest = in_data.shape[1] % self.sentence_length
            if embed is not None:
                in_data = embed(in_data)
            # Strong-Typed-Language fanatics hate this simple trick
            i = 0
            for i in (range((in_data.shape[1] // self.sentence_length) - int(not bool(rest)))):
                info = self.input_encoder(in_data[:, i*self.sentence_length:(i+1)*self.sentence_length])
                h = self.dynamic_memory(info)
            query = self.input_encoder(in_data[:, (i+1)*self.sentence_length:])
            return self.output_module(torch.unsqueeze(query, 2), h)
        return out

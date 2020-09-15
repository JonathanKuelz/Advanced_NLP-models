import torch
import torch.nn as nn

from model.EntNet import InputEncoder


class LSTM(nn.Module):

    def __init__(self, hidden_size, vocab_size, embed_size, device, sentence_size):
        super().__init__()
        self.device = device
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.enc = InputEncoder(sentence_size, embed_size, device)
        self.inner_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.ss = sentence_size

    def forward(self, story, query):
        # story.shape = [Batchsize, max_Story_Length, max_sentence_length]
        # query.shape = [Batchsize, max_query_length]
        bs, mstl, msel = story.shape
        _, ql = query.shape
        padded_query = torch.zeros([bs, 1, self.ss], dtype=torch.long).to(self.device)
        padded_query[:, 0, :ql] = query
        padded_story = torch.zeros([bs, mstl, self.ss], dtype=torch.long).to(self.device)
        padded_story[:, :, :msel] = story
        query_encoded = self.enc(self.embed(padded_query))
        story_encoded = self.enc(self.embed(padded_story))
        empty_in = torch.zeros_like(query_encoded, dtype=torch.float32, device=self.device)
        x = torch.cat([story_encoded, query_encoded, empty_in, empty_in],  dim=1)
        out, (h_n, c_n) = self.inner_lstm(x)
        return self.linear(h_n.squeeze(dim=0))

import torch
import torch.nn as nn


class NLP_LSTM(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim=100, hidden_size=20, sentence_length=10, device='cpu'):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocabulary_size)

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, in_data, embed=None, full=True):
        """
        Args:
            in_data: A sequence of shape batch_size x seq_len x 1
            embed: An embedding function (e.g. torch.nn.embedding)
            full: Determines whether every state or just the last state of the model shall be returned

        Returns: A tensor of shape batchsize x seq_len x embedding dim. if full=False, only the last sequence instead of seq_len

        """
        if embed is not None:
            in_data = embed(in_data)
        x = in_data
        out, (h_n, c_n) = self.lstm(x)
        return self.linear(out) if full else torch.squeeze(self.linear(h_n))

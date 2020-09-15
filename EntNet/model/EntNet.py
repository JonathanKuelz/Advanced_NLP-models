import torch
import torch.nn as nn
import torch.nn.functional as F

""" Define the recurrent entity network"""


class InputEncoder(nn.Module):
    """
    Implemen a simple Input masking as described in Section 2.1 of the paper, Equation 1.
    It is not more than a simple multiplicative mask plus a summation.
    Takes already embedded input.
    Default initialization is Bag of Words (BOW) witha all weights=1. Can be learned though
    """
    def __init__(self, sentence_size, embed_size, device):
        super(InputEncoder, self).__init__()
        self.mask = nn.Parameter(torch.ones([sentence_size, embed_size], dtype=torch.float32, device=device))

    def forward(self, x):
        # x.shape = [Batchsize, max_Story_Length, max_sentence_length, embedding_size]
        # return shape = [Batchsize, max_story_length, embedding_size], summed all words in sentence
        return torch.sum(x * self.mask, 2)

class thres(nn.Threshold):
    def __init__(self, inplace=False):
        super(thres, self).__init__(0., 1., inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


class MemCell(nn.Module):
    def __init__(self, num_blocks, embed_size, keys, activation):
        super().__init__()
        self.num_blocks = num_blocks
        # Where are the keys?
        # Initializers are also "missing"
        self.activation = activation
        self.embed_size = embed_size  # Embedding Size == Units per Block
        self.keys = keys

        self.U = nn.Linear(embed_size, embed_size, bias=True)  # Instead of defining a seperate bias, use the nn one
        self.V = nn.Linear(embed_size, embed_size, bias=False)
        self.W = nn.Linear(embed_size, embed_size, bias=False)
        nn.init.normal_(self.U.weight, 0, 0.1)
        nn.init.normal_(self.V.weight, 0, 0.1)
        nn.init.normal_(self.W.weight, 0, 0.1)
        self.th = thres()

    def get_gate(self, state_j, key_j, inputs):  # Checked with tensorflow, is the same
        """
        Paper equation (2), implements the gate that determines how much a memory should be updated
        """
        a = torch.sum(inputs * state_j, dim=1)
        b = torch.sum(inputs * key_j, dim=1)
        return torch.sigmoid(a + b)

    def get_candidate(self, state_j, key_j, inputs):
        """
        Paper equation (3). The candidate is the new possible value that will be combined with the old state of the cell
        to form the new state.
        """
        key_V = self.V(key_j)
        state_U = self.U(state_j)  # + self.bias
        inputs_W = self.W(inputs)
        return self.activation(state_U + inputs_W + key_V)

    def forward(self, x, state):  # Apart from the different normalization, it's 1:1 tensorflow
        # First, split the hidden state into blocks. U, V and W are shared across blocks
        state = torch.split(state, self.embed_size, 1)
        next_states = []
        for j, state_j in enumerate(state):  # And now, iterate over the blocks, each one of shape [batchsize, embed_size]
            key_j = self.keys[j].unsqueeze(0)
            gate_j = self.get_gate(state_j, key_j, x)
            candidate_j = self.get_candidate(state_j, key_j, x)

            # Equation (4), perform an update on the hidden state
            state_j_next = state_j + gate_j.unsqueeze(-1) * candidate_j
            # Normalize the state, as in equation (5)
            # TODO: In case the norm is very close to 0, the tensorflow repo replaces it by 1!
            state_j_next_norm = torch.abs(torch.norm(state_j_next, p=2, dim=-1, keepdim=True)) + 1e-8  # keepdim because we want to have the norm for each batch
            state_j_next = self.th(state_j_next) / state_j_next_norm

            next_states.append(state_j_next)
        state_next = torch.cat(next_states, dim=1)
        return state_next

    def zero_state(self, bs):
        """Initialize the memory cell to the key values."""
        zero_state = torch.cat([key.unsqueeze(0) for key in self.keys], 1)
        zero_state_batch = zero_state.repeat(bs, 1)
        return zero_state_batch


class OutputModule(nn.Module):
    """An Implementation of the EntNet Output Module as Described in Section 2.3"""
    def __init__(self, num_blocks, vocab_size, embed_size, activation, device):
        super(OutputModule, self).__init__()
        self.activation = activation
        self.num_blocks = num_blocks
        self.embed_size = embed_size
        self.R = nn.Linear(embed_size, vocab_size, bias=False)
        self.H = nn.Linear(embed_size, embed_size, bias=False)
        self.R.weight.data.normal_(0.0, 0.1)
        self.H.weight.data.normal_(0.0, 0.1)

    def forward(self, x, state):
        # State shape = [batchsize, num_blocks, embed_size]
        # x shape = [batchsize, 1, embed_size] -> sums over embed_dims
        state = torch.stack(torch.split(state, self.embed_size, dim=1), dim=1)
        attention = torch.sum(state * x, dim=2)
        attention = attention - torch.max(attention, dim=-1, keepdim=True)[0]
        # attention shape = [Batchsize, Number of Cells] + unsqueezed dimension
        attention = torch.softmax(attention, dim=1).unsqueeze(2)
        u = torch.sum(state * attention, dim=1)
        q = x.squeeze(1)
        y = self.R(self.activation(q + self.H(u)))
        return y


class REN(nn.Module):
    def __init__(self, num_blocks, vocab_size, embed_size, device, sentence_size, query_size):
        super(REN, self).__init__()
        vocab_size = vocab_size + num_blocks  # Add additional vocabulary for keys
        self.device = device
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0).to(device)  # Initial weight is Gaussian (0, 1)
        self.prelu = nn.PReLU(num_parameters=embed_size, init=1.0).to(device)
        self.story_enc = InputEncoder(sentence_size, embed_size, device)
        self.query_enc = InputEncoder(query_size, embed_size, device)

        keys = self.init_keys()
        self.cell = MemCell(num_blocks, embed_size, keys, self.prelu)
        self.output = OutputModule(num_blocks, vocab_size, embed_size, self.prelu, device)
        self.to(device)

    def init_keys(self):  # 1:1 as in tensorflow
        keys = [key for key in torch.arange(self.vocab_size - self.num_blocks, self.vocab_size, device=self.device)]  # dtype=torch.long
        keys = [self.embed(key).squeeze(0) for key in keys]
        return keys

    def forward(self, story, query):
        # story.shape = [Batchsize, max_Story_Length, max_sentence_length]
        # query.shape = [Batchsize, max_query_length]

        story_encoded = self.story_enc(self.embed(story))
        # Unsqueezing the query leads to query of same dimension as story
        query_encoded = self.query_enc(self.embed(query.unsqueeze(1)))
        initial_state = self.cell.zero_state(story.shape[0])  # TODO: Is it okay to always set cell state to zero?
        for i in range(story_encoded.shape[1]):
            initial_state = self.cell(story_encoded[:, i, :], initial_state)
        outputs = self.output(query_encoded, initial_state)
        return outputs

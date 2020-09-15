#!/usr/bin/env python3
# Jonathan Külz and Omar Dahroug, 07/2020

import os

import chainer
import numpy as np
import torch

from Dataset.NLP.NLPTask import NLPTask
from Dataset.NLP.Vocabulary import Vocabulary
from Utils import Visdom


class PTB(NLPTask):
    """
    A custom Dataset Class implementing the Penn Treebank
    Inherits features from NLP task but does not make use of all of them.
    """

    def __init__(self, name=None, seq_len=10):
        super().__init__()

        if name is None:
            print("No name given, implicitly expecting test set")
            name = 'Test'
        elif name.lower() not in ('test', 'train', 'validation'):
            raise ValueError("Invalid name for PTB dataset. Should be test, train or validation.")
        self.name = name
        self.seq_len = seq_len

        # Downloading data, https://docs.chainer.org/en/stable/reference/generated/chainer.datasets.get_ptb_words.html
        # Data Source: https://github.com/wojzaremba/lstm
        self.data = None
        self._get_data()
        self._test_res_win = None  # Visdom plot for test results

        self.ignore_targets = [None]

        # TODO: GloVe Embedding
        # TODO: Should we query sentence-wise?

    def __getitem__(self, idx):

        target = np.asarray(self.data[idx+self.seq_len], dtype=np.int64)
        if target in self.ignore_targets:
            target = np.asarray(0, dtype=np.int64)  # Cross Entropy Loss ignores targets with idx 0
        return {
            "input": np.asarray(np.append(self.data[idx:idx+self.seq_len], [0]), dtype=np.int64),
            "output": target,
            "meta": None
        }

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def _get_data(self):
        """
        Choose the right dataset to iterate over later on. Querying the whole PTB dataset is okay here as it only
        takes a couple kbyte of memory - still, if critical, this could be cached and loaded locally.
        """
        train_data, val_data, test_data = chainer.datasets.get_ptb_words()
        if self.name.lower() == 'train':
            self.data = train_data
        elif self.name.lower() == 'test':
            self.data = test_data
        elif self.name.lower() == 'validation':
            self.data = val_data
        else:
            raise ValueError("Invalid dataset Name")

    def _load_vocabulary(self):
        """Overwrites the NLP load vocabulary to prevent conflicts with other NLP datasets"""
        cache_file = os.path.join(self.cache_dir, "ptb_vocabulary.pth")
        if not os.path.isfile(cache_file):
            print("No Vocabulary found. Creating a new one.")
            word2id = chainer.datasets.get_ptb_words_vocabulary()
            self.vocabulary = Vocabulary.from_dict(word2id)
            self.save_vocabulary()
        else:
            self.vocabulary = torch.load(cache_file)
        self.ignore_targets = [self.vocabulary.words[word] for word in ('<eos>', '<unk>', 'N')]
        return self.vocabulary

    def save_vocabulary(self):
        """Overwrites the NLP save vocabulary to prevent conflicts with other NLP datasets"""
        cache_file = os.path.join(self.cache_dir, "ptb_vocabulary.pth")
        if os.path.isfile(cache_file):
            os.remove(cache_file)
        torch.save(self.vocabulary, cache_file)

    def state_dict(self):
        """Necessary for the state saver"""
        if self._test_res_win is not None:
            return {
                "_test_res_win": self._test_res_win.state_dict(),
                "_test_plot_win": self._test_plot_win.state_dict(),
            }
        else:
            return {}

    def load_state_dict(self, state):
        if state:
            self._ensure_test_wins_exists()
            self._test_res_win.load_state_dict(state["_test_res_win"])
            self._test_plot_win.load_state_dict(state["_test_plot_win"])

    def start_test(self):
        return {}

    def verify_result(self, test, data, net_output):
        _, net_output = net_output.max(-1)

        ref = data["output"]

        mask = 1.0 - ref.eq(0).float()  # 1 if output calculated/desired (target neq 0), 0 else

        correct = (torch.eq(net_output, ref).float() * mask).sum(-1)  # correct outputs, sum over all embedd 0ed words
        total = mask.sum(-1)  # nr of outputs

        correct = correct.data.cpu().numpy()
        total = total.data.cpu().numpy()

        task = "PTB"  # Keep everything in line with the babi pipeline.
        if task not in test:
            test[task] = {"total": 0, "correct": 0}

        d = test[task]
        d["total"] += total
        d["correct"] += correct

    def show_test_results(self, x_value, test, x_label='Epoch'):

        acc_percent = test['PTB']['correct'] / test['PTB']['total'] * 100

        self._ensure_test_wins_exists(xlabel=x_label, ylabel='Accuracy')
        self._test_plot_win.add_point(x_value, acc_percent)

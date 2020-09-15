import tensorflow as tf
import chainer
import numpy as np
from datasets.text_dataset import TextDataset

from embeddings.embedding import GloVeEmbedding


class PTB(TextDataset):
    def __init__(
            self,
            train_batch_size=100,
            valid_batch_size=None,
            test_batch_size=None,
            shuffle_buffer=5000,
            sequence_length=10,
            embedding=None,
            one_hot=True,
            target_vocab_size=None,
            exclude_token=None,
            full_eval=False,
            **kwargs
    ):
        """
        Pentree-Bank data set.

        Args:
            train_batch_size:   training batch size
            valid_batch_size:   validation batch size
            test_batch_size:    test batch size
            shuffle_buffer:     buffer length
            train_data_path:    path to training data txt file
            valid_data_path:    path to validation data txt file
            test_data_path:     path to test data txt file
            sequence_length:    length of sequences to crop from the sentences
            **kwargs:           additional args (currently not used)
        """
        super().__init__("Pentree-Bank",
                         shuffle_buffer=shuffle_buffer,
                         train_batch_size=train_batch_size,
                         valid_batch_size=valid_batch_size,
                         test_batch_size=test_batch_size,
                         embedding=embedding,
                         one_hot=one_hot,
                         sequence_length=sequence_length,
                         target_vocab_size=target_vocab_size,
                         exclude_token=exclude_token,
                         full_eval=full_eval
                         )

        train_data, val_data, test_data = chainer.datasets.get_ptb_words()
        self.ptb_dict_word2ID = chainer.datasets.get_ptb_words_vocabulary()
        self.ptb_dict_ID2word = dict((v, k) for k, v in self.ptb_dict_word2ID.items())
        self.embedding.adapt(self.ptb_dict_word2ID)

        if one_hot:
            self.output_shape = len(self.ptb_dict_ID2word) if target_vocab_size is None else min(target_vocab_size, len(self.ptb_dict_ID2word))
        else:
            self.output_shape = self.embedding.vector_dimension

        self.input_shape = (self.sequence_length, self.embedding.vector_dimension)
        # @todo define class weights
        # self.class_weights = self.compute_class_weights()
        print(f'Dataset input shape: {self.input_shape}, output shape: {self.output_shape}')

        self.test_ds = self.preprocess_data(test_data)
        self.test_ds = self.prepare_generator(self.test_ds, self.test_batch_size)

        self.train_ds = self.preprocess_data(train_data)
        self.valid_ds = self.preprocess_data(val_data)

        self.train_ds = self.prepare_generator(self.train_ds, self.train_batch_size, shuffle=True)
        self.valid_ds = self.prepare_generator(self.valid_ds, self.valid_batch_size)

    def preprocess_data(self, data):
        eos = np.argwhere(data == self.ptb_dict_word2ID["<eos>"])
        eos = np.insert(eos, 0, -1)  # includes first index

        sentences = []
        for idx in range(len(eos) - 1):
            if (eos[idx + 1] - eos[idx] - 1) <= self.sequence_length:  # deletes too short sentences
                continue
            sentences += list([data[eos[idx] + 1: eos[idx + 1]]])

        return tf.data.Dataset.from_generator(
            lambda: self.generator(data=sentences),
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.input_shape, self.output_shape))


if __name__ == '__main__':
    ptb = PTB(
        train_batch_size=7,
        sequence_length=13,
        embedding=GloVeEmbedding(r'./embeddings/GloVe/glove.6B.100d.txt', limit=10000),
        one_hot=True,
        target_vocab_size=111
    )

    x, y = iter(ptb.train_ds).next()
    print(f'Batch shape: {x.shape}, target shape: {y.shape}')
    print(x)
    x, y = iter(ptb.valid_ds).next()
    print(f'Batch shape: {x.shape}, target shape: {y.shape}')
    x, y = iter(ptb.test_ds).next()
    print(f'Batch shape: {x.shape}, target shape: {y.shape}')


    train_data, val_data, test_data = chainer.datasets.get_ptb_words()
    word, count = np.unique(val_data, return_counts=True)
    len(train_data)
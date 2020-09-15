from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
import json


class TextDataset:

    def __init__(self,
                 dataset_name,
                 shuffle_buffer,
                 train_batch_size,
                 valid_batch_size,
                 test_batch_size,
                 embedding,
                 one_hot,
                 sequence_length,
                 target_vocab_size,
                 exclude_token=None,
                 full_eval=False
                 ):
        """
        Abstraction for text datasets.

        Args:
            dataset_name:       name of the dataset
            shuffle_buffer:     size of the buffer
            train_batch_size:
            valid_batch_size:
            test_batch_size:
            embedding:          embedding of the dataset
            one_hot:            one-hot or embedding output
            sequence_length:    length of the input sequence
            target_vocab_size:  target vocabulary size
        """
        self.data_name = dataset_name
        self.shuffle_buffer = shuffle_buffer
        self.train_batch_size = train_batch_size
        self.valid_batch_size = train_batch_size if valid_batch_size is None else valid_batch_size
        self.test_batch_size = train_batch_size if test_batch_size is None else test_batch_size
        self.sequence_length = sequence_length
        self.one_hot = one_hot
        self.exclude_token = exclude_token
        self.target_vocab_size = target_vocab_size
        self.full_eval = full_eval
        self.input_shape = None
        self.output_shape = None

        if embedding is None:
            raise ValueError('No embedding was given')
        self.embedding = embedding
        self.tokenizer = None
        self.class_weights = None

    def tokenize_from_file(self, train_data_path, num_words=None, filters='.?!#$%&()*+-/:;<=>@[\\]^_{|}~\t\n'):
        print('creating tokenizer from file')
        with open(train_data_path, encoding="utf8") as f:
            text = f.read()

        tokenizer = self.build_tokenizer([text], num_words, filters)
        self.embedding.adapt(tokenizer.word_index)

        input_shape = (self.sequence_length, self.embedding.vector_dimension)
        output_shape = len(tokenizer.word_index) if self.one_hot else self.embedding.vector_dimension
        return tokenizer, input_shape, output_shape

    def build_tokenizer(self, sentences: list, num_words=None, filters='.?!#$%&()*+-/:;<=>@[\\]^_{|}~\t\n'):
        """
        Builds tokenizer based on a list of sentences.

        Args:
            sentences:  List of sentences to build the tokenizer from
            num_words:  max number of words used by the tokenizer
            filters:    filters used by the tokenizer

        Returns:
            fitted tokenizer

        """
        tokenizer = Tokenizer(num_words=num_words, filters=filters, oov_token='<OOV>')
        tokenizer.fit_on_texts(sentences)
        return tokenizer

    def prepare_generator(self, data_set, batch_size, shuffle=False):
        """
        Prepares the generator to use for learning.

        Returns:
            BatchDatasets for training, validation, testing

        """
        if shuffle:
            batch_ds = data_set.shuffle(self.shuffle_buffer).batch(batch_size)
        else:
            batch_ds = data_set.batch(batch_size)
        return batch_ds.prefetch(tf.data.experimental.AUTOTUNE)

    def generator(self, data):
        """
        Generates a data point of sequential data of size, used for Wiki103 and PTB
        window_size for a tensorflow dataset. The data point is
        generated with a random start on a random sequence.
        Args:
            data: List of numpy arrays containing sequential data.
        Returns:
            x: Sequential numpy array containing a data point.
            y: Ground truth of x
        """
        idxs = np.arange(len(data))
        for idx in idxs:
            sentence = tf.constant(data[idx], dtype=tf.int32)
            start_indexes = self.get_start_indexes(sentence)

            if start_indexes is None:
                continue

            if not self.full_eval:
                start_indexes = [np.random.choice(start_indexes)]

            for seq_start in start_indexes:
                x = sentence[seq_start: seq_start + self.sequence_length]
                y = sentence[seq_start + self.sequence_length]
                try:
                    if self.one_hot:
                        yield self.embedding(x), tf.one_hot(y, depth=self.output_shape)
                    else:
                        yield self.embedding(x), self.embedding(y)
                except Exception as e:
                    print(x)
                    raise e

    def get_start_indexes(self, sentence):
        """
        Determines valid start index.

        Args:
            sentence: sentence to get a start value for

        Returns:
            valid index or None if there is no valid index
        """
        sentence = sentence.numpy()

        if len(sentence) - self.sequence_length <= 0:
            return None

        if self.exclude_token is not None:
            sentence[np.isin(sentence, self.exclude_token)] = np.iinfo(np.int32).max

        if self.target_vocab_size is None:   # all targets are valid
            possible_index = np.arange(0, len(sentence) - self.sequence_length)
            mask = sentence[:-self.sequence_length] < np.iinfo(np.int32).max
            if mask.sum() == 0:
                return None
            possible_index = possible_index[mask]
        else:
            possible_index = (np.arange(len(sentence))[(sentence < self.target_vocab_size)]) \
                             - self.sequence_length
            possible_index = possible_index[possible_index > 0]
            if len(possible_index) == 0:
                return None
        return possible_index

    def dump_tokenizer(self, path):
        with open(path, 'w') as file:
            json.dump(self.tokenizer.to_json(), file)

    def load_tokenizer(self, path):
        print('loading tokenizer from json')
        with open(path, 'r') as file:
            json_string = json.load(file)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        self.embedding.adapt(tokenizer.word_index)

        input_shape = (self.sequence_length, self.embedding.vector_dimension)
        output_shape = len(tokenizer.word_index) if self.one_hot else self.embedding.vector_dimension
        return tokenizer, input_shape, output_shape

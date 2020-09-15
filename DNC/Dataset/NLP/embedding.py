import numpy as np
import tensorflow as tf


class GloVeEmbedding:

    def __init__(self, path, limit=None):
        """
        A class to read the word embedding file and to create the word embedding matrix.
        This model assumes the data to be a multidimensional numpy array.

        Args:
            path:       Path to the token-embedding space mapping file
            limit:      limiting the loaded embedding
        """
        self.glove_embedding_path = path
        self.glove_embeddings_dict = self.load_glove_embedding(limit=limit)
        self.embedding_weights = None
        self.vocab_length = None
        self.vector_dimension = len(self.glove_embeddings_dict[list(self.glove_embeddings_dict.keys())[0]])
        self.min_token = 0

    def adapt(self, word_index):
        """
        Adapts the embedding to a specific tokenizer.

        Args:
            tokenizer:  tokenizer to adapt to

        Returns:
            None

        """
        embeddings_dict = {}
        for word, token in word_index.items():
            embeddings_dict[token] = self.glove_embeddings_dict.get(word, np.zeros(self.vector_dimension))
        keys = list(embeddings_dict.keys())
        keys.sort()
        self.min_token = keys[0]
        embedding_weights = np.array([embeddings_dict[key] for key in keys])
        # embedding_weights = tf.constant_initializer(embedding_weights)
        self.embedding_weights = tf.compat.v1.get_variable(
            name='embedding_weights',
            # shape=(self.vocab_length, self.vector_dimension),
            initializer=embedding_weights,
            trainable=False)

    def load_glove_embedding(self, limit=None):
        """
        Creates the mapping from file.

        Returns:
            Dict containing the word -> embedding space mapping

        """
        embeddings_index = {}
        with open(self.glove_embedding_path, encoding='utf8') as f:
            for vocab_length, line in enumerate(f):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
                if limit is not None and vocab_length >= limit:
                    break
            self.vocab_length = vocab_length
        return embeddings_index

    @tf.function
    def __call__(self, sequence):
        return tf.nn.embedding_lookup(self.embedding_weights, sequence - self.min_token)


if __name__ == '__main__':
    class Tokenizer:
        def __init__(self):
            self.word_index = {'the': 1, 'car': 2, 'tree': 3, 'cat': 4, 'run': 5}


    embedding = GloVeEmbedding(r'./embeddings/GloVe/glove.6B.100d.txt')
    embedding.adapt(Tokenizer().word_index)
    embedding.glove_embeddings_dict['cat']
    embedding.__call__()



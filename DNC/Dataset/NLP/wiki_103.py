import tensorflow as tf
from embeddings.embedding import GloVeEmbedding
from datasets.text_dataset import TextDataset


class Wiki103(TextDataset):
    def __init__(self,
                 train_batch_size=100,
                 valid_batch_size=None,
                 test_batch_size=None,
                 shuffle_buffer=5000,
                 train_data_path=None,
                 valid_data_path=None,
                 test_data_path=None,
                 sequence_length=10,
                 embedding=None,
                 one_hot=True,
                 target_vocab_size=None,
                 tokenizer_path=None,
                 exclude_token=None,
                 full_eval=False,
                 **kwargs):
        """
        Wikitext-103 data set.

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

        super().__init__("Wikitext-103",
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

        if tokenizer_path is None:
            self.tokenizer, self.input_shape, self.output_shape = self.tokenize_from_file(train_data_path)
        else:
            self.tokenizer, self.input_shape, self.output_shape = self.load_tokenizer(tokenizer_path)

        if target_vocab_size is not None:
            self.output_shape = target_vocab_size

        if test_data_path is not None:
            test_ds = self.load_data(test_data_path)
            self.test_ds = self.prepare_generator(test_ds, self.test_batch_size)

        if train_data_path is not None:
            train_ds = self.load_data(train_data_path)
            # @todo define class weights member
            # self.class_weights = self.compute_class_weights()
            self.train_ds = self.prepare_generator(train_ds, self.train_batch_size, shuffle=True)

        if valid_data_path is not None:
            valid_ds = self.load_data(valid_data_path)
            self.valid_ds = self.prepare_generator(valid_ds, self.valid_batch_size)

    def load_data(self, file_path):
        """
        Loads data from file system.

        Args:
            file_path:     path to training data txt file

        Returns:
            a list of
        """
        with open(file_path, encoding="utf8") as f:
            data = list(map(lambda line: line.strip(), f.readlines()))
        data = [line for line in data if line != '' and line[0] != '=']

        data = self.tokenizer.texts_to_sequences(data)
        dataset = tf.data.Dataset.from_generator(
            lambda: self.generator(data=data),
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.input_shape, self.output_shape))
        return dataset


if __name__ == '__main__':
    train_path = r'./data/Wikitext_103/wikitext-103-v1/wikitext-103/wiki.train.tokens'
    valid_path = r'./data/Wikitext_103/wikitext-103-v1/wikitext-103/wiki.valid.tokens'
    test_path = r'./data/Wikitext_103/wikitext-103-v1/wikitext-103/wiki.test.tokens'

    wiki103 = Wiki103(train_data_path=train_path,
                      valid_data_path=valid_path,
                      test_data_path=test_path,
                      train_batch_size=128,
                      sequence_length=10,
                      embedding=GloVeEmbedding(r'./embeddings/GloVe/glove.6B.100d.txt', limit=5000),
                      one_hot=True,
                      target_vocab_size=10000
                      )

    x, y = iter(wiki103.train_ds).next()
    print(f'Batch shape: {x.shape}, target shape{y.shape}')
    x, y = iter(wiki103.valid_ds).next()
    print(f'Batch shape: {x.shape}, target shape{y.shape}')
    x, y = iter(wiki103.test_ds).next()
    print(f'Batch shape: {x.shape}, target shape{y.shape}')

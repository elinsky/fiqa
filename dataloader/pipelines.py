import tensorflow as tf
import tensorflow_datasets as tdfs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer


class AspectOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, data_config):
        self.one_hot_encoder = OneHotEncoder()
        self.aspect_labels = data_config.aspect_labels

    def fit(self, X, y=None):
        self.one_hot_encoder.fit(self.aspect_labels)
        return self

    def transform(self, X, y=None):
        # https://github.com/tensorflow/tensorflow/issues/12851#issuecomment-669863983
        headlines, aspects, sentiments = [tf.data.Dataset.from_tensor_slices(list(x)) for x in zip(*X)]

        # Convert tensorflow aspects back to list of strings
        aspects_str = list(map(tf.compat.as_str_any, tdfs.as_numpy(aspects)))
        aspects_list = [[aspect] for aspect in aspects_str]

        # Transform
        aspects_one_hot = self.one_hot_encoder.transform(aspects_list).toarray()

        # Package back up
        aspects_one_hot_tf = tf.convert_to_tensor(aspects_one_hot, dtype=tf.int32)
        aspects_dataset = tf.data.Dataset.from_tensor_slices(aspects_one_hot_tf, name='aspect')
        transformed = tf.data.Dataset.zip((headlines, aspects_dataset, sentiments))

        return transformed


class HeadlineTFIDFTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, data_config):
        self.vocab_size = data_config.vocab_size
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')

    def fit(self, X, y=None):
        # Unpack dataset https://github.com/tensorflow/tensorflow/issues/12851#issuecomment-669863983
        headlines, aspects, sentiments = [tf.data.Dataset.from_tensor_slices(list(x)) for x in zip(*X)]

        # Convert headlines back to list of string
        headlines_str = list(map(tf.compat.as_str_any, tdfs.as_numpy(headlines)))

        # Tokenize
        self.tokenizer.fit_on_texts(headlines_str)

        return self

    def transform(self, X, y=None):
        # Unpack dataset https://github.com/tensorflow/tensorflow/issues/12851#issuecomment-669863983
        headlines, aspects, sentiments = [tf.data.Dataset.from_tensor_slices(list(x)) for x in zip(*X)]

        # Convert headlines back to list of string
        headlines_str = list(map(tf.compat.as_str_any, tdfs.as_numpy(headlines)))

        # Tokenize
        headlines_tfidf = self.tokenizer.texts_to_matrix(headlines_str, mode='tfidf')

        # Package back up
        headlines_tf = tf.convert_to_tensor(headlines_tfidf, dtype=tf.float32)
        headlines_dataset = tf.data.Dataset.from_tensor_slices(headlines_tf, name='headlines_tfidf')
        transformed = tf.data.Dataset.zip((headlines_dataset, aspects, sentiments))
        return transformed

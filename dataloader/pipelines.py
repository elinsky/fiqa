import tensorflow as tf
import tensorflow_datasets as tdfs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class AspectOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, data_config):
        self.one_hot_encoder = OneHotEncoder()
        self.aspect_labels = data_config.aspect_labels

    def fit(self, X, y=None):
        self.one_hot_encoder.fit(self.aspect_labels)

    def transform(self, X, y=None):
        # Unpack dataset
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

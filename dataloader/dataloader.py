import json
from typing import Tuple

import tensorflow as tf


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config) -> tf.data.Dataset:
        """Loads dataset from path"""

        # Load data from file
        f = open(data_config.path)
        data = json.load(f)
        headlines = []
        sentiments = []
        aspects = []
        for sample in data.values():
            headlines.append(sample['sentence'])
            # If there are actually 2 labels for a document, only take the first
            sentiments.append(float(sample['info'][0]['sentiment_score']))
            cleaned_l1_aspect, cleaned_l2_aspect = DataLoader._clean_aspect(sample['info'][0]['aspects'])
            aspect = cleaned_l1_aspect + ', ' + cleaned_l2_aspect
            aspects.append(aspect)
        f.close()

        # Create TensorFlow dataset
        headlines_tf = tf.data.Dataset.from_tensor_slices(headlines, name='headlines')
        aspects_tf = tf.data.Dataset.from_tensor_slices(aspects, name='aspects')
        sentiments_tf = tf.data.Dataset.from_tensor_slices(sentiments, name='sentiments')
        dataset = tf.data.Dataset.zip((headlines_tf, aspects_tf, sentiments_tf))

        return dataset

    @staticmethod
    def _clean_aspect(aspect: str) -> Tuple[str, str]:
        """
        Given an aspect, return only the first two levels. Discard level 3+.
        E.g. '[\'Corporate/Sales/Deal\']' -> ('Corporate', 'Sales')
        :param aspect:
        :return:
        """
        cleaned_aspect = aspect.strip('][\'').split('/')
        level_1 = cleaned_aspect[0]
        level_2 = cleaned_aspect[1]
        return level_1, level_2

    @staticmethod
    def _preprocess_data(dataset, batch_size, buffer_size):
        """Preprocess and split into training, validation, and test sets"""
        # TODO
        pass

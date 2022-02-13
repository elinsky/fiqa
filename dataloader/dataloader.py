import json
from typing import Tuple

import tensorflow as tf
from sklearn.pipeline import Pipeline


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
    def preprocess_data(dataset, data_config, pipeline: Pipeline):
        """Preprocess and split into training, validation, and test sets"""

        # Shuffle and split
        dataset = dataset.shuffle(data_config.buffer_size, data_config.seed)
        train_dataset, rest = DataLoader._split_dataset(dataset, data_config.train_split)
        validation_dataset, test_dataset = DataLoader._split_dataset(rest, data_config.validation_split / (
                1.0 - data_config.train_split))

        # Preprocess training data
        pipeline.fit(train_dataset)
        train_dataset = pipeline.transform(train_dataset)

        # Preprocess validation and test data
        validation_dataset = pipeline.transform(validation_dataset)
        test_dataset = pipeline.transform(test_dataset)

        return train_dataset, validation_dataset, test_dataset

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
    def _split_dataset(dataset: tf.data.Dataset, split_perc: float) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Splits a dataset into two. The first dataset returned has split_perc rows.
        """
        n = len(dataset)
        n_first = int(n * split_perc)
        first_dataset = dataset.take(n_first)
        second_dataset = dataset.skip(n_first)

        return first_dataset, second_dataset

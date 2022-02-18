import json
from typing import Tuple

import tensorflow as tf
from sklearn.pipeline import Pipeline


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(file_path) -> tf.data.Dataset:
        """Loads dataset from path. Does as minimal pre-processing as possible."""

        # Load data from file
        f = open(file_path)
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
    def preprocess_data(train_dataset, test_dataset, data_config, pipeline: Pipeline):
        """Preprocess and split into training, validation, and test sets"""

        # Shuffle and split train dataset
        train_shuffled = train_dataset.shuffle(data_config.buffer_size, data_config.seed,
                                               reshuffle_each_iteration=False)
        train_split, val_split = DataLoader._split_dataset(train_shuffled, data_config.train_split)

        # Fit pipeline on training data then transform training data
        pipeline.fit(train_split)
        train_split_transformed = pipeline.transform(train_split)

        # Transform validation and test data
        val_split_transformed = pipeline.transform(val_split)
        test_split_transformed = pipeline.transform(test_dataset)

        return train_split_transformed, val_split_transformed, test_split_transformed

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

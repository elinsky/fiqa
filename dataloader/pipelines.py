import typing

import tensorflow as tf
import tensorflow_datasets as tdfs
import transformers
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class AspectOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, data_config):
        self.one_hot_encoder = OneHotEncoder()
        self.aspect_labels = data_config.aspect_labels

    def fit(self, X, y=None):
        self.one_hot_encoder.fit(self.aspect_labels)
        return self

    def transform(self, X, y=None) -> tf.data.Dataset:
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

    def transform(self, X, y=None) -> tf.data.Dataset:
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


def tokenize_sentences(sentences: typing.List[str], tokenizer: PreTrainedTokenizerBase) \
        -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Given a list of sentences and a tokenizer, this tokenizes the sentences
    and returns a tuple of input_ids, token_type_ids, and attention_mask as
    tensors.
    """
    encoded_dict = tokenizer(
        sentences,
        padding=True,  # Padding = True will pad to the longest sequence in the batch.
        return_tensors='tf',
        max_length=128,  # We truncate to 128 tokens.
        return_attention_mask=True,
        truncation=True)

    # Unpack encoded dict
    # input_ids (batch size, 128). Integer encoded sentences with padding on the end.
    # token_type_ids (batch size, 128). This appears to be used when you need to pass into a downstream model multiple
    # sentences. E.g. question answering. You want to pass in two sentences (say a context sentence. and a question).
    # And you want to indicate to the model which sentence is which.
    # attention_mask (batch_size, 128). This is 1s for the real tokens, and 0s for the padded tokens. This tells the
    # downstream model which tokens to 'attend' to, and which it can ignore.
    input_ids = encoded_dict['input_ids']
    token_type_ids = encoded_dict['token_type_ids']
    attention_mask = encoded_dict['attention_mask']

    tf.cast(attention_mask, dtype=tf.float32)

    return input_ids, token_type_ids, attention_mask


class BertHeadlineTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, data_config):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Unpack dataset https://github.com/tensorflow/tensorflow/issues/12851#issuecomment-669863983
        headlines, aspects, sentiments = [tf.data.Dataset.from_tensor_slices(list(x)) for x in zip(*X)]
        # Convert headlines back to list of string
        headlines_str = list(map(tf.compat.as_str_any, tdfs.as_numpy(headlines)))
        # Tokenize
        input_ids, token_type_ids, attention_mask = tokenize_sentences(headlines_str, self.tokenizer)

        # Package back up
        input_ids_dataset = tf.data.Dataset.from_tensor_slices(input_ids, name='input_ids')
        token_type_ids_dataset = tf.data.Dataset.from_tensor_slices(token_type_ids, name='token_type_ids')
        attention_mask_dataset = tf.data.Dataset.from_tensor_slices(attention_mask, name='attention_mask')
        headlines_dataset = tf.data.Dataset.zip((input_ids_dataset, token_type_ids_dataset, attention_mask_dataset))
        transformed_dataset = tf.data.Dataset.zip((headlines_dataset, aspects, sentiments))

        return transformed_dataset


class DistilBertHeadlineTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, data_config):
        self.tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
            './configs/reuters_tokenizer')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Unpack dataset https://github.com/tensorflow/tensorflow/issues/12851#issuecomment-669863983
        headlines, aspects, sentiments = [tf.data.Dataset.from_tensor_slices(list(x)) for x in zip(*X)]
        # Convert headlines back to list of string
        headlines_str = list(map(tf.compat.as_str_any, tdfs.as_numpy(headlines)))
        # Tokenize
        # Distilbert does not have token_type_ids
        output = self.tokenizer(headlines_str, pad_to_max_length=True).data
        input_ids, attention_mask = output['input_ids'], output['attention_mask']

        # Package back up
        input_ids_dataset = tf.data.Dataset.from_tensor_slices(input_ids, name='input_ids')
        attention_mask_dataset = tf.data.Dataset.from_tensor_slices(attention_mask, name='attention_mask')
        headlines_dataset = tf.data.Dataset.zip((input_ids_dataset, attention_mask_dataset))
        transformed_dataset = tf.data.Dataset.zip((headlines_dataset, aspects, sentiments))

        return transformed_dataset

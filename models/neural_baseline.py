import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Dense
from transformers import TFBertModel

from dataloader.dataloader import DataLoader
from dataloader.pipelines import AspectOneHotEncoder, BertHeadlineTokenizer
from executor.trainer import Trainer
from utils.logger import get_logger
from .base_model import BaseModel
# internal
from .hierarchical_classifier import HierarchicalClassifier

LOG = get_logger('Neural Baseline')


class NeuralBaselineModel(tf.keras.Model):

    def __init__(self):
        super(NeuralBaselineModel, self).__init__(name='NeuralBaselineModel')
        self.aspect_classifier = HierarchicalClassifier(4, 12, 2, 4, 9)
        self.sentiment_regression = Dense(units=1, activation=None, use_bias=True)

        # In order to save the model, the bert layer needs to be a subclass of tf.keras.layers.Layer. Hence, the need to
        # use the .bert attribute, which is a huggingface 'MainLayer' object and has the @keras_serializable decorator.
        # https://github.com/huggingface/transformers/blob/0f69b924fbda6a442d721b10ece38ccfc6b67275/src/transformers/models/bert/modeling_tf_bert.py#L696
        self.bert = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).bert
        assert isinstance(self.bert, tf.keras.layers.Layer)

    def call(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs
        bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)
        x = bert_outputs.last_hidden_state  # (batch size, 28, 768)

        attention_mask = tf.expand_dims(attention_mask, 2)  # (batch size, 28) -> (batch size, 28, 1).
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)

        se = x * attention_mask  # (batch_size, 28, 768) * (batch_size, 28, 1) -> (batch_size, 28, 768)
        den = tf.math.reduce_sum(attention_mask, axis=1)  # (batch_size, 28, 1) -> (batch size, 1)
        se = tf.math.reduce_sum(se, axis=1) / den  # (batch_size, 28, 768) -> (batch size, 768)

        aspect_prob = self.aspect_classifier(se)  # Probability distribution over L2 classes (batch size, 27)
        sentiment_logits = self.sentiment_regression(se)  # (batch size, 1)

        return sentiment_logits, aspect_prob


class NeuralBaseline(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.data_config = self.config.data
        self.epochs = self.config.train.epochs
        self.batch_size = self.config.train.batch_size
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.learning_rate = self.config.train.learning_rate

    def load_datasets(self):
        """Loads and preprocesses data and splits into train, validate, and test sets"""
        LOG.info(f'Loading {self.config.data.train_path} dataset...')
        train_dataset = DataLoader().load_data(self.config.train_path)
        LOG.info(f'Loading {self.config.data.test_path} dataset...')
        test_dataset = DataLoader().load_data(self.config.test_path)
        pre_process_steps = [('Aspect One Hot Encoder', AspectOneHotEncoder(self.data_config)),
                             ('BERT Headline Tokenizer', BertHeadlineTokenizer(self.data_config))]
        pipe = Pipeline(pre_process_steps)
        self.train_dataset, self.val_dataset, self.test_dataset = DataLoader.preprocess_data(train_dataset,
                                                                                             test_dataset,
                                                                                             self.data_config, pipe)

    def build(self):
        """Builds the model"""
        self.model = NeuralBaselineModel()
        LOG.info('Keras Model was built successfully')

    def train(self):
        """Compiles and trains the model"""
        LOG.info('Training started')
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        aspect_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        sentiment_loss_fn = tf.keras.losses.MeanSquaredError()

        def loss_fn(aspect_labels, aspect_predictions, sentiment_labels, sentiment_predictions):
            return aspect_loss_fn(aspect_labels, aspect_predictions) + \
                   sentiment_loss_fn(sentiment_labels, sentiment_predictions)

        train_mse_metric = tf.keras.metrics.MeanSquaredError(name='Train Sentiment MSE')
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='Train Aspect Accuracy')
        val_mse_metric = tf.keras.metrics.MeanSquaredError(name='Validation Sentiment MSE')
        val_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='Validation Aspect Accuracy')

        trainer = Trainer(self.model, self.train_dataset.batch(self.batch_size),
                          self.val_dataset.batch(self.batch_size), loss_fn, optimizer, train_acc_metric,
                          train_mse_metric, val_acc_metric, val_mse_metric, self.epochs)
        trainer.train()

    def evaluate(self):
        """Evaluate the trained model on the test dataset."""
        aspect_predictions = []
        sentiment_predictions = []
        aspect_labels = []
        sentiment_labels = []

        for headline, aspect_label, sentiment_label in self.test_dataset.batch(1):
            # Make prediction
            sentiment_pred, aspect_pred = self.model(headline)
            # Get most probable class label
            aspect_pred = np.argmax(aspect_pred)
            # Get actual class label
            aspect_label = np.argmax(aspect_label)

            sentiment_predictions.append(float(sentiment_pred))
            sentiment_labels.append(float(sentiment_label))
            aspect_predictions.append(aspect_pred)
            aspect_labels.append(aspect_label)

        print(classification_report(aspect_labels, aspect_predictions))
        print('Aspect 2 Macro F1 Score:', f1_score(aspect_labels, aspect_predictions, average='macro'))

        print('Sentiment MSE:', mean_squared_error(sentiment_labels, sentiment_predictions))
        print('Sentiment R^2:', r2_score(sentiment_labels, sentiment_predictions))

        # TODO calc F1 and error for aspect 1

        return aspect_predictions, sentiment_predictions, aspect_labels, sentiment_labels

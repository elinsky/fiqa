import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Dense

from dataloader.dataloader import DataLoader
from dataloader.pipelines import AspectOneHotEncoder, HeadlineTFIDFTokenizer
from executor.trainer import Trainer
from utils.logger import get_logger
from .base_model import BaseModel
# internal
from .hierarchical_classifier import HierarchicalClassifier

LOG = get_logger('Naive Baseline')


class NaiveBaselineModel(tf.keras.Model):

    def __init__(self):
        super(NaiveBaselineModel, self).__init__()
        self.aspect_classifier = HierarchicalClassifier(4, 12, 2, 4, 9)
        self.sentiment_regression = Dense(units=1, activation=None, use_bias=True)

    def call(self, inputs):
        sentiment_logits = self.sentiment_regression(inputs)  # (batch size, 1)
        aspect_prob = self.aspect_classifier(inputs)  # Probability distribution over L2 classes (batch size, 27)
        return sentiment_logits, aspect_prob


class NaiveBaseline(BaseModel):

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

    def load_data(self):
        """Loads and preprocesses data"""
        LOG.info(f'Loading {self.config.data.path} dataset...')
        dataset = DataLoader().load_data(self.config.data)
        pre_process_steps = [('Aspect One Hot Encoder', AspectOneHotEncoder(self.data_config)),
                             ('TF-IDF Headline Tokenizer', HeadlineTFIDFTokenizer(self.data_config))]
        pipe = Pipeline(pre_process_steps)
        self.train_dataset, self.val_dataset, self.test_dataset = DataLoader.preprocess_data(dataset, self.data_config,
                                                                                             pipe)

    def build(self):
        """Builds the model"""
        self.model = NaiveBaselineModel()
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
        """Predicts results for the test set"""
        aspect_predictions = []
        sentiment_predictions = []
        aspect_labels = []
        sentiment_labels = []

        for headline, aspect_label, sentiment_label in self.val_dataset.batch(1):
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
        print('aspect level 2 Macro F1 score:', f1_score(aspect_labels, aspect_predictions, average='macro'))

        print('sentiment MSE:', mean_squared_error(sentiment_labels, sentiment_predictions))
        print('sentiment R^2:', r2_score(sentiment_labels, sentiment_predictions))

        # TODO calc F1 and error for aspect 1

        return aspect_predictions, sentiment_predictions, aspect_labels, sentiment_labels

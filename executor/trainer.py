import datetime
import os

import tensorflow as tf

from utils.logger import get_logger

LOG = get_logger('trainer')


class Trainer:

    def __init__(self, model, train_dataset, val_dataset, loss_fn, optimizer, train_aspect_metric,
                 train_sentiment_metric, val_aspect_metric, val_sentiment_metric, epochs):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_aspect_metric = train_aspect_metric
        self.train_sentiment_metric = train_sentiment_metric
        self.train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.val_aspect_metric = val_aspect_metric
        self.val_sentiment_metric = val_sentiment_metric
        self.val_loss_metric = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        self.epochs = epochs

        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, './tf_ckpts', max_to_keep=3)

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = 'logs/gradient_tape/' + self.model.name + '/' + self.current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_log_dir = 'logs/gradient_tape/' + self.model.name + '/' + self.current_time + '/val'
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

        self.model_save_path = 'saved_models/'

    def train_step(self, batch):
        headlines, aspect_labels, sentiment_labels = batch
        with tf.GradientTape() as tape:
            sentiment_predictions, aspect_predictions = self.model(headlines)
            step_loss = self.loss_fn(aspect_labels, aspect_predictions, sentiment_labels, sentiment_predictions)

        grads = tape.gradient(step_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_aspect_metric.update_state(aspect_labels, aspect_predictions)
        self.train_sentiment_metric.update_state(sentiment_labels, sentiment_predictions)
        self.train_loss_metric.update_state(step_loss)

        return step_loss, aspect_predictions, sentiment_predictions

    def train(self):
        for epoch in range(self.epochs):
            LOG.info(f'Start epoch {epoch}')

            for step, training_batch in enumerate(self.train_dataset):
                step_loss, aspect_predictions, sentiment_predictions = self.train_step(training_batch)
                LOG.info("Loss at step %d: %.2f" % (step, step_loss))

            train_acc = self.train_aspect_metric.result()
            LOG.info("Training aspect accuracy over epoch: %.4f" % (float(train_acc)))
            train_mse = self.train_sentiment_metric.result()
            LOG.info("Training sentiment MSE over epoch: %.4f" % (float(train_mse)))
            train_loss = self.train_loss_metric.result()
            LOG.info("Training loss over epoch: %.4f" % (float(train_loss)))

            self._write_train_summary(train_loss, epoch)

            save_path = self.checkpoint_manager.save()
            LOG.info("Saved checkpoint: {}".format(save_path))

            # reset training metrics every epoch
            self.train_aspect_metric.reset_states()
            self.train_sentiment_metric.reset_states()
            self.train_loss_metric.reset_states()

            # Run a validation loop at the end of each epoch
            self.test(epoch)

        save_path = os.path.join(self.model_save_path, 'naive_baseline/1/')
        # tf.saved_model.save(self.model, save_path) # TODO

    def test_step(self, batch):
        # can use for both validation and test
        headlines, aspect_labels, sentiment_labels = batch
        sentiment_predictions, aspect_predictions = self.model(headlines)
        step_loss = self.loss_fn(aspect_labels, aspect_predictions, sentiment_labels, sentiment_predictions)
        self.val_aspect_metric.update_state(aspect_labels, aspect_predictions)
        self.val_sentiment_metric.update_state(sentiment_labels, sentiment_predictions)
        self.val_loss_metric.update_state(step_loss)

        return step_loss, aspect_predictions, sentiment_predictions

    def test(self, epoch):
        # can use for both validation and test
        for val_step, val_batch in enumerate(self.val_dataset):
            _, val_aspect_predictions, val_sentiment_predictions = self.test_step(val_batch)

        val_acc = self.val_aspect_metric.result()
        LOG.info("Validation aspect accuracy over epoch: %.4f" % (float(val_acc)))
        val_mse = self.train_sentiment_metric.result()
        LOG.info("Validation sentiment MSE over epoch: %.4f" % (float(val_mse)))
        val_loss = self.val_loss_metric.result()
        LOG.info("Validation loss over epoch: %.4f" % (float(val_loss)))

        self._write_val_summary(val_loss, epoch)

        # reset validation metrics every epoch
        self.val_aspect_metric.reset_states()
        self.val_sentiment_metric.reset_states()
        self.val_loss_metric.reset_states()

    def _write_train_summary(self, loss, epoch):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('train loss', loss, step=epoch)
            tf.summary.scalar('train aspect accuracy', self.train_aspect_metric.result(), step=epoch)
            tf.summary.scalar('train sentiment MSE', self.train_sentiment_metric.result(), step=epoch)
            # tensorboard --logdir logs/gradient_tape

    def _write_val_summary(self, loss, epoch):
        with self.val_summary_writer.as_default():
            tf.summary.scalar('val loss', loss, step=epoch)
            tf.summary.scalar('val aspect accuracy', self.val_aspect_metric.result(), step=epoch)
            tf.summary.scalar('val sentiment MSE', self.val_sentiment_metric.result(), step=epoch)

import tensorflow as tf
from tensorflow.keras.layers import Dense


class HierarchicalClassifier(tf.keras.layers.Layer):

    def __init__(self, n_l1, n_corp, n_econ, n_market, n_stock):
        super(HierarchicalClassifier, self).__init__()
        self.n_l1 = n_l1
        self.n_corp = n_corp
        self.n_econ = n_econ
        self.n_market = n_market
        self.n_stock = n_stock
        self.level_1_classifier = Dense(units=n_l1, activation=None, use_bias=True)
        self.level_2_corporate_classifier = Dense(units=n_corp, activation=None, use_bias=True)
        self.level_2_economy_classifier = Dense(units=n_econ, activation=None, use_bias=True)
        self.level_2_market_classifier = Dense(units=n_market, activation=None, use_bias=True)
        self.level_2_stock_classifier = Dense(units=n_stock, activation=None, use_bias=True)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = inputs.shape[0]

        # Calc logits for individual classifiers
        level_1_logits = self.level_1_classifier(inputs)  # (batch size, n_level_1)
        level_2_corporate_logits = self.level_2_corporate_classifier(inputs)  # (batch size, n_corporate)
        level_2_economy_logits = self.level_2_economy_classifier(inputs)  # (batch size, n_economy)
        level_2_market_logits = self.level_2_market_classifier(inputs)  # (batch size, n_market)
        level_2_stock_logits = self.level_2_stock_classifier(inputs)  # (batch size, n_stock)

        # Calc probabilities for individual classifiers
        level_1_prob = tf.nn.softmax(level_1_logits)
        level_2_corporate_prob = tf.nn.softmax(level_2_corporate_logits)
        level_2_economy_prob = tf.nn.softmax(level_2_economy_logits)
        level_2_market_prob = tf.nn.softmax(level_2_market_logits)
        level_2_stock_prob = tf.nn.softmax(level_2_stock_logits)

        # Calc output probabilities
        # TODO - tf.expand_dims(level_2_corporate_prob, 1) # expand here instead. That way you keep the shape more consistently as (batch size, N_LEVEL_1, level 2)
        level_2_corporate_prob = tf.expand_dims(level_2_corporate_prob, -1)  # (batch size, n_corporate, 1)
        zeros = tf.zeros(shape=(batch_size, self.n_corp, self.n_l1 - 1))
        level_2_corporate_prob = tf.concat([level_2_corporate_prob, zeros], -1)  # (batch size, n_corporate, n_level_1)

        level_2_economy_prob = tf.expand_dims(level_2_economy_prob, -1)
        zeros_top = tf.zeros(shape=(batch_size, self.n_econ, self.n_l1 - 1 - 2))
        zeros_bottom = tf.zeros(shape=(batch_size, self.n_econ, self.n_l1 - 1 - 1))
        level_2_economy_prob = tf.concat([zeros_top, level_2_economy_prob, zeros_bottom], -1)

        level_2_market_prob = tf.expand_dims(level_2_market_prob, -1)
        zeros_top = tf.zeros(shape=(batch_size, self.n_market, self.n_l1 - 1 - 1))
        zeros_bottom = tf.zeros(shape=(batch_size, self.n_market, self.n_l1 - 1 - 2))
        level_2_market_prob = tf.concat([zeros_top, level_2_market_prob, zeros_bottom], -1)

        level_2_stock_prob = tf.expand_dims(level_2_stock_prob, -1)
        zeros = tf.zeros(shape=(batch_size, self.n_stock, self.n_l1 - 1))
        level_2_stock_prob = tf.concat([zeros, level_2_stock_prob], -1)

        composite = tf.concat([level_2_corporate_prob, level_2_economy_prob, level_2_market_prob, level_2_stock_prob],
                              1)  # (batch size, n_level_2, n_level_1)
        composite_t = tf.linalg.matrix_transpose(composite)  # (batch size, n_level_1, n_level_2)

        level_1_prob = tf.expand_dims(level_1_prob, 1)

        aspect_pred = tf.linalg.matmul(level_1_prob, composite_t)  # This is a softmax over all level 2 classes
        aspect_pred = tf.squeeze(aspect_pred, axis=1)  # (batch size, n_level_2)

        return aspect_pred

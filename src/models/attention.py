import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        self.score_dense.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        score = self.score_dense(x)
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(x * weights, axis=1)
    
    def get_config(self):
        return super().get_config()

def build_attention_model(vectorizer, vocab_size, embedding_dim, lstm_units, dropout_rate, num_classes):
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = AttentionPooling()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="goemotions_attention")

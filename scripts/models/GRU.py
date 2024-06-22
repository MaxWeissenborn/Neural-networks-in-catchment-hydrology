import tensorflow as tf
import tensorflow.keras.backend as K  # for custom loss function
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from tensorflow.keras.models import Model


description = "GRU Model, 1 GRU Layer and 1 dropout layer."

parameterDict = {"T": [1, 300, int],
                 "hu": [10, 500, int],
                 "dropout": [0.05, 0.5, float]}


def create_model(T, hu, dropout, D, lr):

    def custom_loss(y_true, y_pred):

        alpha = K.std(y_pred) / K.std(y_true)
        beta = K.sum(y_pred) / K.sum(y_true)  # no need to calc mean
        r = tfp.stats.correlation(y_true, y_pred, sample_axis=None, event_axis=None)
        return K.sqrt(K.square(1 - r) + K.square(1 - alpha) + K.square(1 - beta))

    i = Input(shape=(T, D))
    x = GRU(hu)(i)
    x = Dropout(dropout)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(i, out)

    model.compile(
        loss=custom_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    )

    return model

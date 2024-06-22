import tensorflow as tf
import tensorflow.keras.backend as K  # for custom loss function
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Model

description = "CNN-Model, 3 convolution layers and 1 global-max-pooling layer."

parameterDict = {"T": [50, 300, int],
                 "f1": [100, 500, int],
                 "ks1": [3, 9, int]}


def create_model(T, D, f1, ks1, lr):
    def custom_loss(y_true, y_pred):
        alpha = K.std(y_pred) / K.std(y_true)
        beta = K.sum(y_pred) / K.sum(y_true)  # no need to calc mean
        r = tfp.stats.correlation(y_true, y_pred, sample_axis=None, event_axis=None)
        return K.sqrt(K.square(1 - r) + K.square(1 - alpha) + K.square(1 - beta))

    i = Input(shape=(T, D))
    x = Conv1D(f1, ks1, activation='relu', padding="same")(i)
    x = MaxPooling1D(2)(x)
    x = Conv1D(f1 * 2, ks1, activation='relu', padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(f1 * 4, ks1, activation='relu', padding="same")(x)
    x = GlobalMaxPooling1D()(x)
    out = Dense(1, activation="LeakyReLU")(x)
    
    model = Model(i, out)

    model.compile(
        loss=custom_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    )

    return model

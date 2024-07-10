import numpy as np
import tensorflow as tf


class ExponentialSoftmaxPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        # Calculate the numerator: sum(y * exp_y) along the second dimension (axis=1)
        max_value = tf.reduce_max(inputs, axis=1, keepdims=True)
        numerator = tf.reduce_sum(inputs * tf.exp(inputs - max_value), axis=1)
        # Calculate the denominator: sum(exp_y) along the second dimension (axis=1)
        denominator = tf.reduce_sum(tf.exp(inputs - max_value), axis=1)
        # Perform the pooling operation: (sum(y * exp_y)) / (sum(y))
        result = numerator / denominator
        return result


def create_model(x_data_original, y_data_original):
    num_outputs = y_data_original.shape[1]
    input_shape = np.shape(x_data_original[0])

    model = tf.keras.Sequential()
    if x_data_original.ndim == 2:
        model.add(tf.keras.layers.Dense(num_outputs, input_shape=input_shape, activation='sigmoid'))
    elif x_data_original.ndim == 3:
        def create_dense_layer(nodes):
            return tf.keras.layers.Dense(units=nodes, activation='sigmoid')

        model.add(tf.keras.layers.TimeDistributed(create_dense_layer(num_outputs), input_shape=input_shape))
        model.add(ExponentialSoftmaxPooling())
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train, x_val, y_val, log_dir):
    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor for early stopping ('val_loss' or 'val_accuracy', etc.)
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True,  # Restores the model weights from the epoch with the best validation performance
        start_from_epoch=50,  # Train for 50 epochs save, warmStart
        min_delta=0.1,  # minimum improvement
        mode='auto',  # stop when accuracy stops increasing
        verbose=0
    )

    # save results in tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    history = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_data=(x_val, y_val),
                        callbacks=[tensorboard, early_stopping], verbose=0)
    return history

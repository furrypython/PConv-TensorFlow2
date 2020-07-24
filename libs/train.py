import os
from datetime import datetime
import tensorflow as tf
import numpy as np

from process_data import create_input_dataset
from loss import loss_total


@tf.function
def train_step(model, example, batch_size, vgg16, optimizer):
    inputs, targets = create_input_dataset(example, batch_size)

    with tf.GradientTape() as tape:
        #output = model(inputs, training=True)
        loss = loss_total(model, inputs, targets, vgg16, training=True)

    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


@tf.function
def val_step(model, example, batch_size, vgg16):
    inputs, targets = create_input_dataset(example, batch_size)
    loss = loss_total(model, inputs, targets, vgg16, training=False)
    return loss


def fit(model, input_data, batch_size, epochs, validation_data,
        steps_per_epoch, validation_steps, vgg16, optimizer, save_dir):

    train_loss_results = []
    val_loss_results = []

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch, ))
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()

        for step in range(steps_per_epoch):
            example = next(input_data)
            loss_train = train_step(model, example, batch_size, vgg16,
                                    optimizer)
            # Update training metric.
            epoch_loss_avg.update_state(loss_train)

        # Log every epoch.
        train_loss_results.append(epoch_loss_avg.result())

        # Run a validation loop at the end of each epoch.
        for step_val in range(validation_steps):
            example_val = next(validation_data)
            loss_val = val_step(model, example_val, batch_size, vgg16)
            epoch_val_loss_avg.update_state(loss_val)
        # Log every epoch.
        val_loss_results.append(epoch_val_loss_avg.result())

        print("Training loss: %.4f, Validation loss: %.4f" %
              (epoch_loss_avg.result(), epoch_val_loss_avg.result()))

        # Reset metrics at the end of each epoch
        epoch_loss_avg.reset_states()
        epoch_val_loss_avg.reset_states()

    # Save weights file.
    datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    checkpoint_name = save_dir + '/epoch-' + str(
        epoch) + '-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.h5'
    model.save_weights(checkpoint_name, overwrite=True, save_format='h5')

    return {'loss': train_loss_results, 'val_loss': val_loss_results}

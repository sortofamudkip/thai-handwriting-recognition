import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import cgan2 as cgan
from ..load_data import load_images

def load_and_preprocess_real_data_for_cgan(dataset_path: Path, batch_size: int, image_size: int):
    ## Load data
    train_dataset, validation_dataset, test_dataset, class_names = load_images('./processed_dataset/train', './processed_dataset/test', 32, label_mode="categorical")
    # scale the pixel values to be between 0 and 1 and reverse the pixel values (black is 1, white is 0)
    train_dataset = train_dataset.map(lambda x, y: ((1-(x/255.0)), y))
    validation_dataset = validation_dataset.map(lambda x, y: ((1-(x/255.0)), y))
    test_dataset = test_dataset.map(lambda x, y: ((1-(x/255.0)), y))

    full_train_dataset = train_dataset.concatenate(validation_dataset)

    ## create a tf dataset with the real images + labels are X and 1s are y
    real_images = []
    real_labels = []
    for image_batch, labels_batch in full_train_dataset:
        real_images.append(image_batch)
        real_labels.append(labels_batch)
    real_images = np.concatenate(real_images).astype("float32")
    real_labels = np.concatenate(real_labels).astype("float32")
    # convert real_labels to the same dimension as the images (64x64x1)
    expanded_labels = np.expand_dims(real_labels, axis=1)
    expanded_labels = np.expand_dims(expanded_labels, axis=1)
    expanded_labels = np.tile(expanded_labels, (1, 64, 64, 1))
    real_images_and_labels = np.concatenate([real_images, expanded_labels], axis=3)
    # create a dataset
    real_dataset = tf.data.Dataset.from_tensor_slices((real_images_and_labels, np.ones(real_images.shape[0])))

    return real_dataset

def create_fake_data_for_cgan(generator, num_images=1000):
    batch_size = 32 # doesn't have to be 32
    fake_images = []
    fake_labels = []
    for i in range(num_images):
        noise = np.random.normal(size=(batch_size, 128))
        labels = np.eye(42)[np.random.choice(42, batch_size)]
        noise_and_labels = np.concatenate([noise, labels], axis=1)
        generated_images = generator.predict(noise_and_labels, verbose=0)
        fake_images.append(generated_images)
        fake_labels.append(labels)
    fake_images = np.concatenate(fake_images).astype("float32")
    fake_labels = np.concatenate(fake_labels).astype("float32")
    # convert fake_labels to the same dimension as the images (64x64x1)
    expanded_labels = np.expand_dims(fake_labels, axis=1)
    expanded_labels = np.expand_dims(expanded_labels, axis=1)
    expanded_labels = np.tile(expanded_labels, (1, 64, 64, 1))
    fake_images_and_labels = np.concatenate([fake_images, expanded_labels], axis=3)
    fake_dataset = tf.data.Dataset.from_tensor_slices((fake_images_and_labels, np.zeros(fake_images.shape[0])))
    return fake_dataset

def pretrain_discriminator(discriminator, real_dataset, fake_dataset, epochs=10):
    # usage: pretrain_discriminator(gan.discriminator, real_dataset, fake_dataset, epochs=10)
    # compile the model
    discriminator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()]
    )
    # train the model
    history = discriminator.fit(
        real_dataset.concatenate(fake_dataset).shuffle(1000).batch(32),
        epochs=epochs,
        verbose=2
    )
    return history, discriminator
    
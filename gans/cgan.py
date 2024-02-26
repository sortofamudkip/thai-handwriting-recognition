# import tf and keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

class CGAN(keras.Model):
    def __init__(self, latent_dim, num_classes=42, generator=None, discriminator=None):
        super(CGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # constants
        self.num_image_channels = 1 # * because the images are grayscale
        self.num_classes = num_classes
        self.image_size = 64

        # some hyperparameters
        self.latent_dim = latent_dim

        self.generator = self.get_generator() if generator is None else generator
        self.discriminator = self.get_discriminator() if discriminator is None else discriminator

        # define loss function
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False) # real or fake loss

        # loss trackers
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    def one_hot_labels_to_image(self, labels):
        expanded_labels = tf.expand_dims(labels, axis=1)
        expanded_labels = tf.expand_dims(expanded_labels, axis=1)
        expanded_labels = tf.tile(expanded_labels, (1, self.image_size, self.image_size, 1))
        return expanded_labels

    def get_generator(self):
        input_layer = keras.layers.Input(shape=(self.latent_dim+self.num_classes,))
        x = keras.layers.Dense(4*4*64, activation="relu")(input_layer)
        x = keras.layers.Reshape((4, 4, 64))(x)
        x = keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
        x = keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
        x = keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
        x = keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
        x = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
        return keras.models.Model(inputs=input_layer, outputs=x)

    def get_discriminator(self): # num_channels=1 because the images are grayscale
        images_and_labels = keras.layers.Input(shape=(64, 64, self.num_image_channels+self.num_classes))
        x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(images_and_labels)
        x = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x = keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
        x = keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1, activation="sigmoid")(x)
        return keras.models.Model(inputs=images_and_labels, outputs=x)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(CGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.discriminator.compile(loss=self.loss_fn, optimizer=self.d_optimizer)
        self.generator.compile(loss=self.loss_fn, optimizer=self.g_optimizer)
        
    def generate_fake_samples_without_labels(self, n_samples, labels):
        noise = tf.random.normal((n_samples, self.latent_dim))
        noise_and_labels = tf.concat([noise, labels], axis=1)
        fake_samples = self.generator(noise_and_labels)
        return fake_samples

    def generate_fake_labels(self, batch_size) -> tf.Tensor:
        # create one-hot encoded labels
        labels = tf.random.uniform([batch_size], 0, self.num_classes, dtype=tf.int32)
        return tf.one_hot(labels, self.num_classes)
    
    def train_step(self, data):
        # & Unpack the real data.
        X_real, y_real = data
        batch_size = tf.shape(X_real)[0]

        d_loss = self.train_D(X_real, y_real, batch_size)
        g_loss = self.train_G(batch_size)

        # & Update loss
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def train_D(self, X_real, y_real, batch_size: int):
        # & Generate fake samples
        fake_labels = self.generate_fake_labels(batch_size)
        fake_samples = self.generate_fake_samples_without_labels(batch_size, fake_labels)
        fake_samples_and_labels = tf.concat([fake_samples, self.one_hot_labels_to_image(fake_labels)], axis=3)
        # & concatenate real samples and labels
        real_samples_and_labels = tf.concat([X_real, self.one_hot_labels_to_image(y_real)], axis=3)
        # & combine real and fake samples
        all_samples_and_labels = tf.concat([real_samples_and_labels, fake_samples_and_labels], axis=0)
        # & create labels that say "these are real images" and "these are fake images"
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        all_labels = tf.concat([real_labels, fake_labels], axis=0)
        # & Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(all_samples_and_labels)
            d_loss = self.loss_fn(all_labels, predictions)

        # & update the discriminator
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        return d_loss
    
    def train_G(self, batch_size): # doesn't have to be batch_size but it's a good default
        # & create labels that say "these are real images" 
        misleading_labels = tf.zeros((batch_size, 1))
        
        # & Train the generator
        with tf.GradientTape() as tape:
            # & Generate fake samples and labels
            fake_labels = self.generate_fake_labels(batch_size)
            fake_samples = self.generate_fake_samples_without_labels(batch_size, fake_labels)
            fake_samples_and_labels = tf.concat([fake_samples, self.one_hot_labels_to_image(fake_labels)], axis=3)
            # & obtain the discriminator's predictions
            predictions = self.discriminator(fake_samples_and_labels)
            # & calculate the loss and update the generator
            g_loss = self.loss_fn(misleading_labels, predictions) # * we want the discriminator to be wrong
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return g_loss
    
    def load_model(self, model_dir: Path):
        self.generator = keras.models.load_model(model_dir / "generator")
        self.discriminator = keras.models.load_model(model_dir / "discriminator")

# define a callback to save the models after the final epoch
class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, model: CGAN, output_dir: Path):
        super().__init__()
        self.model = model
        self.output_dir = output_dir

    def on_train_end(self, logs=None):
        print(f"Saving models to {self.output_dir}...")
        self.model.generator.save(self.output_dir / "generator.h5")
        self.model.discriminator.save(self.output_dir / "discriminator.h5")

# define a callback to generate an image after each epoch
class GenerateImageCallback(keras.callbacks.Callback):
    def __init__(self, generator, latent_dim: int, num_classes: int, output_dir: Path, frequency: int = 1):
        super().__init__()
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.frequency = frequency

    # create the "output" directory if it doesn't exist
    def on_train_begin(self, logs=None):
        self.output_dir.mkdir(exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            n_samples = 5
            labels = tf.one_hot([i for i in range(5)], self.num_classes)
            fake_samples = self.generator.predict(tf.concat([tf.random.normal((n_samples, self.latent_dim)), labels], axis=1))
            fig, axs = plt.subplots(1, n_samples, figsize=(n_samples, 1))
            for i in range(n_samples):
                axs[i].imshow(fake_samples[i, :, :, 0], cmap="gray")
                axs[i].axis("off")
            plt.savefig(self.output_dir / f"generated_image_{epoch}.png")
            plt.close()
            plt.savefig(self.output_dir / f"generated_image_{epoch}.png")
            plt.close()


if __name__ == "__main__":
    from ..load_data import load_images

    output_dir = Path("output")
    ## Load data
    train_dataset, validation_dataset, test_dataset, class_names = load_images('./processed_dataset/train', './processed_dataset/test', 32, label_mode="categorical")

    # create an instance of the CGAN
    cgan = CGAN(latent_dim=128, num_classes=42)

    # compile the model
    cgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.9),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.9),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=False)
    )    # train the model

    # train the model
    cgan.fit(validation_dataset,
            epochs=3,
            callbacks=[
                GenerateImageCallback(cgan.generator, 128, 42, output_dir, frequency=1),
                SaveModelCallback(cgan, output_dir)
            ]
    ) # validation dataset for now, because it's smaller
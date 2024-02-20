from tensorflow import keras
from pathlib import Path

# define an autoencoder model that takes 64x64x1 images and returns 64x64x1 images
def build_autoencoder_model(autoencoder_latent_dim:int, input_shape=(64, 64, 1)):
    input_img = keras.layers.Input(shape=input_shape)
    # Encoder
    x = keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = keras.layers.Flatten()(x)
    encoded = keras.layers.Dense(autoencoder_latent_dim, activation="relu")(x)

    # Decoder
    x = keras.layers.Dense(8*8*16, activation="relu")(encoded)
    x = keras.layers.Reshape((8, 8, 16))(x)
    x = keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2DTranspose(8, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)


    # get the autoencoder model
    autoencoder = keras.Model(input_img, x)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()

    # also get the encoder model
    encoder = keras.Model(input_img, encoded)

    return encoder, autoencoder

def train_model(
        model: keras.Model,
        train_images,
        validation_images,
        model_path: Path,
        epochs=15,
        batch_size=64,
    ) -> keras.callbacks.History:

    # # create a callback to stop training if the validation accuracy does not decrease for 3 epochs
    # early_stopping = keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=3,
    #     restore_best_weights=True
    # )
    # create a callback to save the model on the best validation accuracy
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath= model_path/f'model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
    )

    # train the autoencoder
    history = model.fit(
        x=train_images,
        y=train_images,
        validation_data=(validation_images, validation_images),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[model_checkpoint],
    )

    return history
if __name__ == "__main__":
    encoder, autoencoder = build_autoencoder_model(128)
from tensorflow import keras
from pathlib import Path

# define an autoencoder model that takes 64x64x1 images and returns 64x64x1 images
def build_autoencoder_model(input_shape=(64, 64, 1), learning_rate=1e-3) -> keras.Model:
    input_img = keras.layers.Input(shape=input_shape)
    
    # Encoder
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input_img)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    encoded = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    
    # Decoder
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoded)
    x = keras.layers.UpSampling2D((2, 2))(x)
    
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    
    decoded = keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
    
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.MeanSquaredError(),
    )
    autoencoder.summary()

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

    # create a callback to stop training if the validation accuracy does not decrease
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    # create a callback to save the model on the best validation accuracy
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath= Path(model_path)/f'model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
    )
    # create a callback to reduce the learning rate if the validation accuracy does not decrease
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )

    # train the autoencoder
    history = model.fit(
        x=train_images,
        y=train_images,
        validation_data=(validation_images, validation_images),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[model_checkpoint,early_stopping,reduce_lr],
        verbose=2
    )

    return history


if __name__ == "__main__":
    encoder, autoencoder = build_autoencoder_model(128)
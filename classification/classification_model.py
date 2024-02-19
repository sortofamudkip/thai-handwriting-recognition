import numpy as np
from tensorflow import keras
from pathlib import Path

DATA_AUGMENTATION_LAYERS = [
    keras.layers.RandomZoom(-0.2),
    keras.layers.RandomTranslation(0.1, 0.1),
]

def data_augmentation(images):
    for layer in DATA_AUGMENTATION_LAYERS:
        images = layer(images)
    return images


def create_classification_model(input_shape, num_classes, is_use_augmentation):
    inputs = keras.Input(shape=input_shape)
    # * rescale the input
    x = keras.layers.Rescaling(1./255)(inputs)
    # * apply data augmentation
    if is_use_augmentation:
        x = data_augmentation(x)
    # * convolutional layers
    x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(128, 3, activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    # * fully connected layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.10)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    # * output layer (softmax activation function for multiclass classification)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

def train_model(
        model: keras.Model,
        train_dataset,
        validation_dataset,
        model_path: Path,
        epochs=15,
    ) -> keras.callbacks.History:

    # compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    # create a callback to stop training if the validation accuracy does not improve for 3 epochs
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        patience=3,
        restore_best_weights=True
    )
    # create a callback to save the model on the best validation accuracy
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath= model_path/f'model.h5',
        monitor='val_sparse_categorical_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
    )

    # train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint]
    )
    return history

def load_model(model_path):
    model = create_classification_model((64, 64, 1), 42, False)
    model.load_weights(model_path)
    return model

if __name__ == '__main__':
    pass
    # create the model
    # model = create_model((64, 64, 1), len(class_names)) # class_names is 42
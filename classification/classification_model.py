import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from .cnn_models import basic_model, medium_model, big_model, medium_filter5_model, medium_dropout50_model, mobilenet

# in the model, the images are first normalised to the range [0, 1], then augmented, so fill_value=1 (white) is used
DATA_AUGMENTATION_LAYERS = [
    keras.layers.RandomZoom((-0.3,0.1), (-0.3,0.1), fill_mode='constant',fill_value=1), 
    keras.layers.RandomTranslation((-0.2, 0.2), (-0.2, 0.2), fill_mode="constant", fill_value=1),
    keras.layers.RandomRotation((-0.03, 0.03)),
]

def data_augmentation(images):
    for layer in DATA_AUGMENTATION_LAYERS:
        images = layer(images)
    return images

MODELS = {
    'basic': basic_model.create_classification_model,
    'medium': medium_model.create_classification_model,
    'medium_filter5': medium_filter5_model.create_classification_model, # 5x5 filter size
    'big': big_model.create_classification_model,
    'medium_dropout50': medium_dropout50_model.create_classification_model,
    'mobilenet': mobilenet.create_classification_model,
}

def train_model(
        model: keras.Model,
        train_dataset,
        validation_dataset,
        model_path: Path,
        epochs:int=15,
        learning_rate:float=0.001,
    ) -> keras.callbacks.History:

    # compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    # create a callback to stop training if the validation accuracy does not improve for 3 epochs
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=16,
        restore_best_weights=True
    )
    # create a callback to save the model on the best validation accuracy
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath= model_path/f'model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
    )

    # train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint],
        verbose=2,
    )
    return history

def mobilenet_prepare_input(image_dataset: tf.data.Dataset):
    # convert to rgb
    image_dataset = image_dataset.map(lambda image, label: (tf.image.grayscale_to_rgb(image), label))
    # resize to 224x224
    image_dataset = image_dataset.map(lambda image, label: (tf.image.resize(image, (224, 224)), label))
    # run the remaining preprocessing steps for the mobilenet_v2 model
    preprocess_input = keras.applications.mobilenet_v2.preprocess_input
    image_dataset = image_dataset.map(lambda image, label: (preprocess_input(image), label))
    return image_dataset

def get_classification_model(model_name, num_classes, use_augmentation=False):
    global MODELS
    if model_name not in MODELS:
        raise ValueError(f"Invalid model name: {model_name}")
    augmentation_layers = data_augmentation if use_augmentation else None
    return MODELS[model_name]((64, 64, 1), num_classes, augmentation_layers)

def load_model(model_name, model_path, use_augmentation=False):
    model = get_classification_model(model_name, 42, use_augmentation)
    model.load_weights(model_path)
    return model

if __name__ == '__main__':
    pass
    # create the model
    # model = create_model((64, 64, 1), len(class_names)) # class_names is 42
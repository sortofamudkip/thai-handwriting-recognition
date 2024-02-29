from tensorflow import keras
import keras_tuner as kt

def build_classification_model(
        input_shape, 
        num_classes,
        augmentation_layers:'function',
        hp_bool_batch_norm:bool,
        hp_dropout_rate:float,
        hp_learning_rate:float,
        hp_first_dense_units:int,
        second_dense_units:int,
        hp_num_conv_layers:int,
    ):
    # & create the model
    inputs = keras.Input(shape=input_shape)
    # * rescale the input
    x = keras.layers.Rescaling(1./255)(inputs)
    # * apply data augmentation
    if augmentation_layers:
        x = augmentation_layers(x)
    # * convolutional layers
    for filter_size in [32, 64, 128, 256][:hp_num_conv_layers]:
        x = keras.layers.Conv2D(filter_size, 3, activation='relu')(x)
        if hp_bool_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Dropout(hp_dropout_rate)(x)
    # * flatten layer
    x = keras.layers.Flatten(name="flatten")(x)
    # * fully connected layers
    x = keras.layers.Dense(hp_first_dense_units, activation='relu')(x)
    x = keras.layers.Dropout(hp_dropout_rate)(x)
    x = keras.layers.Dense(second_dense_units, activation='relu')(x)
    # * output layer (softmax activation function for multiclass classification)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    # & create the model
    model = keras.Model(inputs, outputs)

    # & compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    return model

DATA_AUGMENTATION_LAYERS = [
    keras.layers.RandomZoom((-0.3,0.1), (-0.3,0.1), fill_mode='constant',fill_value=1), 
    keras.layers.RandomTranslation((-0.2, 0.2), (-0.2, 0.2), fill_mode="constant", fill_value=1),
    keras.layers.RandomRotation((-0.03, 0.03)),
]

def data_augmentation(images):
    for layer in DATA_AUGMENTATION_LAYERS:
        images = layer(images)
    return images

def build_hp_model(hp: kt.HyperParameters):
    # define hyperparameters
    hp_bool_batch_norm = hp.Boolean('batch_norm') # total 2 values
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.7, step=0.1) # total 7 values
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5]) # total 3 values
    hp_first_dense_units = hp.Choice('first_dense_units', values=[128, 256, 512]) # total 3 values
    second_dense_units = int(hp_first_dense_units/2)
    hp_num_conv_layers = hp.Choice('num_conv_layers', values=[3, 4]) # total 2 values

    # build the model
    model = build_classification_model(
        (64, 64, 1), 
        44, 
        data_augmentation, 
        hp_bool_batch_norm, 
        hp_dropout_rate, 
        hp_learning_rate, 
        hp_first_dense_units, 
        second_dense_units, 
        hp_num_conv_layers
    )
    return model



if __name__ == "__main__":
    pass
    # # example usage
    # tuner = build_tuner((64, 64, 1), 42, None)
    # tuner.search(train_dataset, validation_dataset, epochs=15, verbose=2)
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # best_model = tuner.hypermodel.build(best_hps)
    # best_model.summary()
    # # * train the best model
    # best_model.fit(train_dataset, validation_data=validation_dataset, epochs=100, verbose=2)
    # # * try saving the best model
    # try:
    #     best_model.save('best_model.h5')
    # except Exception as e:
    #     print("Error saving the best model")
    #     print(e)
    # # * try saving the best model weights
    # try:
    #     best_model.save_weights('best_model_weights.h5')
    # except Exception as e:
    #     print("Error saving the best model weights")
    #     print(e)
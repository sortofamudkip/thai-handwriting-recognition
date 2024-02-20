from tensorflow import keras

def create_classification_model(input_shape, num_classes, augmentation_layers:'function'=None):

    inputs = keras.Input(shape=input_shape)
    # * rescale the input
    x = keras.layers.Rescaling(1./255)(inputs)
    # * apply data augmentation if provided
    if augmentation_layers:
        x = augmentation_layers(x)
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

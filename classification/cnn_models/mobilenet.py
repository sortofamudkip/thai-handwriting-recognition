from tensorflow import keras

def create_classification_model(input_shape, num_classes, augmentation_layers:'function'=None):

    base_mobilenet_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_mobilenet_model.trainable = False
    # create a new model on top
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_mobilenet_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model
from tensorflow import keras

"""
From the paper:
"To show the usefulness of our proposed dataset for research in Thai handwriting
recognition, we selected three popular CNN architectures: CNN with four convolutional
layers, LeNet‑5, and VGG‑13 with batch normalization. The results are shown in Table 4 to
benchmark the proposed dataset. All classifiers were repeated five times by shuffling the
training set and averaging accuracy on the test set. The hyper‑parameters used to train all
the models were batch size 32, dropout 0.5, epoch 100, and optimizer Adam. The testing
results show that VGG‑13 with BN outperforms the others in terms of accuracy."
"""

def create_classification_model(input_shape, num_classes, augmentation_layers:'function'=None):
    inputs = keras.Input(shape=input_shape)
    # * rescale the input
    x = keras.layers.Rescaling(1./255)(inputs)
    # * apply data augmentation
    if augmentation_layers:
        x = augmentation_layers(x)
    # * convolutional layers
    for filter_size, dropout_ratio in zip([32, 64, 128, 256], [0.5, 0.5, 0.5, 0.5]):
        x = keras.layers.Conv2D(filter_size, 3, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Dropout(dropout_ratio)(x)
    # * flatten layer
    x = keras.layers.Flatten()(x)
    # * fully connected layers
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    # * output layer (softmax activation function for multiclass classification)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)

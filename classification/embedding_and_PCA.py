import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from sklearn.decomposition import PCA

from tensorflow import keras

from load_data import load_images
import classification.classification_model as c_model
import classification.evaluation as c_eval
import classification.visualisation as c_vis
import gans.wcgan_gp as wcgan

# boilerplate for installing thai font
import matplotlib.font_manager as fm
font_path = 'fonts/Ayuthaya.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

def embeddings_PCA_mobilenet():
    # load mobilenetv2 model (NOT trainable) and weights and print summary
    model = keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    model.trainable = False
    model.summary()

    # load images
    train_dataset, validation_dataset, test_dataset, class_names = load_images('./processed_dataset/train', './processed_dataset/test', 32, image_size=(224, 224))
    # concatenate datasets
    all_dataset = train_dataset.concatenate(validation_dataset).concatenate(test_dataset)

    # since our images are grayscale, we need to convert them to 3 channels
    def convert_to_rgb(image, label):
        return tf.image.grayscale_to_rgb(image), label

    all_dataset = all_dataset.map(convert_to_rgb)

    # also scale the images to the range [0, 1]
    def scale_image(image, label):
        return image / 255, label

    all_dataset = all_dataset.map(scale_image)

    # obtain embeddings
    embeddings = model.predict(all_dataset)
    all_labels = np.concatenate([y for x, y in all_dataset], axis=0) 
    all_embeddings = embeddings.reshape(embeddings.shape[0], -1)

    # get PCA of the entire dataset
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(all_embeddings)


    # plot the PCA of the entire dataset
    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=all_labels, cmap='tab10', s=5, alpha=0.5)
    # legend1 = ax.legend(*scatter.legend_elements(), title='Classes')
    # ax.add_artist(legend1)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA of embeddings (MobilenetV2)') 

    fig, ax = plt.subplots()
    for label in (0, 1):
        ax.scatter(pca_embeddings[all_labels == label, 0], pca_embeddings[all_labels == label, 1], label=class_names[label], s=10,)
    ax.legend()
    ax.set_title('PCA(2) of MobileNetV2 embeddings')


def embeddings_PCA_medium_model():
    batch_size = 32
    ## Load data
    train_dataset, validation_dataset, test_dataset, class_names = load_images('./processed_dataset/train', './processed_dataset/test', 
                                                                            batch_size, label_mode="int")
    # create the model
    model = c_model.get_classification_model('medium', len(class_names), use_augmentation=True)

    # set the weights
    model.load_weights('output/classification/classification-medium19382704_🥑🍍🍠🥪/model.h5')
    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    # make model NOT trainable
    model.trainable = False
    model.summary()
    # just to make sure it's working: evaluate the model on the test set
    loss, accuracy = model.evaluate(test_dataset)
    print("Accuracy", accuracy)

    # to get image embeddings: remove the everything after the flatten layer
    flatten_layer = model.get_layer('flatten_1')
    embedding_model = keras.models.Model(inputs=model.inputs, outputs=flatten_layer.output)
    embedding_model.summary()

    # concatenate the entire dataset
    all_data = train_dataset.concatenate(validation_dataset).concatenate(test_dataset)
    # get the embeddings
    all_embeddings = embedding_model.predict(all_data)

    # get the labels
    all_labels =  np.concatenate([y for x, y in all_data], axis=0)

    ## PCA of the embeddings
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)
    all_embeddings_pca = pca.transform(all_embeddings)

    fig1, ax1 = plt.subplots()
    for label in (0, 1):
        ax1.scatter(all_embeddings_pca[all_labels == label, 0], all_embeddings_pca[all_labels == label, 1], label=class_names[label], s=10)
    ax1.legend()
    ax1.set_title('PCA of embeddings (Specialised model)')
    ax1.set_xlabel('PCA 1')
    ax1.set_ylabel('PCA 2')


    fig1, ax1 = plt.subplots()
    for label in (0, 1): # first two letters
        ax1.scatter(all_embeddings_pca[all_labels == label, 0], all_embeddings_pca[all_labels == label, 1], label=class_names[label], s=10)
    ax1.legend()
    ax1.set_title('PCA of embeddings (Specialised model)')
    ax1.set_xlabel('PCA 1')
    ax1.set_ylabel('PCA 2')
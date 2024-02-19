import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path


# load the data using tf.keras.utils.image_dataset_from_directory
def load_images(images_path_train, images_path_test, batch_size, image_size=(64, 64)):
    seed = 123
    batch_size = batch_size
    # class_names is the list of thai characters
    THAI_ALPHABET = 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
    class_names = list(THAI_ALPHABET)
    # get the train and validation dataset
    train_dataset, validation_dataset = keras.preprocessing.image_dataset_from_directory(
        Path(images_path_train),
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        color_mode='grayscale',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        subset="both",
        seed=seed,
        validation_split=0.2,
    )
    # get the test dataset
    test_dataset = keras.preprocessing.image_dataset_from_directory(
        Path(images_path_test),
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        color_mode='grayscale',
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    return train_dataset, validation_dataset, test_dataset, class_names

if __name__ == '__main__':
    # load the data
    train_dataset, validation_dataset, test_dataset, class_names = load_images('./processed_dataset/train', './processed_dataset/test', 32)
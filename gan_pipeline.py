# import tf and keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from load_data import load_images
from job_utils import create_output_dir
from gans.cgan2 import CGAN, GenerateImageCallback, SaveModelCallback
import logging, json
from classification.classification_model import MODELS

# boilerplate for installing thai font
import matplotlib.font_manager as fm
font_path = str(Path(__file__).parent / 'fonts/NotoSerifThai-Regular.ttf')
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


def run_pipeline(
    pipeline_name: str,
    dataset_path: Path,
    gan_num_epochs:int,
    gan_D_learning_rate:float,
    gan_G_learning_rate:float,
    gan_batch_size:int,
    gan_latent_dim:int,
    use_pretrained_D:bool,

    classification_model_name: str,
        
):
    output_dir = create_output_dir(pipeline_name, "gan_jobs", skip_if_exists=False)
    output_dir_gan = output_dir / 'gan'
    output_file_name = str((output_dir / f"log.log").resolve())

    # log and print the parameters
    paramaters = {
        "dataset_path": str(dataset_path),
        "gan": {
            "num_epochs": gan_num_epochs,
            "D_learning_rate": gan_D_learning_rate,
            "G_learning_rate": gan_G_learning_rate,
            "batch_size": gan_batch_size,
            "latent_dim": gan_latent_dim,
            "use_pretrained_D": use_pretrained_D,
        },
        "classifier": {
            "model_name": classification_model_name,
        }
    }
    logging.info(f"Parameters:\n{json.dumps(paramaters, indent=2)}")
    print(f"Parameters:\n{json.dumps(paramaters, indent=2)}")

    # determine class_names from dataset_path
    class_names = None
    if dataset_path.name == 'processed_dataset':
        class_names = 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
    elif dataset_path.name == 'processed_dataset_binary':
        class_names = 'กฮ'
    else:
        raise ValueError(f"Invalid dataset path: {dataset_path}")
    print(f"Dataset path name: {dataset_path.name}, class_names: {class_names}")

    # load the data
    train_dataset, validation_dataset, test_dataset, class_names = load_images(dataset_path/'train', dataset_path/'test', gan_batch_size, label_mode="categorical", class_names=class_names)
    # scale the pixel values to be between 0 and 1 and reverse the pixel values (black is 1, white is 0)
    train_dataset = train_dataset.map(lambda x, y: (1-(x/255.0), y))
    validation_dataset = validation_dataset.map(lambda x, y: (1-(x/255.0), y))
    test_dataset = test_dataset.map(lambda x, y: (1-(x/255.0), y))

    full_train_dataset = train_dataset.concatenate(validation_dataset)

    # if use_pretrained_D is True, load the pretrained model
    discriminator = None
    if use_pretrained_D:
        pretrained_D_path = Path(__file__).parent / 'gans' / 'pretrained_discriminator.h5'
        discriminator = keras.models.load_model(pretrained_D_path)


    # create an instance of the CGAN
    cgan = CGAN(latent_dim=gan_latent_dim, num_classes=len(class_names), discriminator=discriminator)

    # compile the model
    cgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=gan_D_learning_rate, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=gan_G_learning_rate, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=False)
    )    # train the model

    # train the model
    cgan.fit(
        full_train_dataset,
        epochs=gan_num_epochs,
        callbacks=[
            GenerateImageCallback(cgan.generator, gan_latent_dim, len(class_names), output_dir_gan, frequency=1),
            SaveModelCallback(cgan, output_dir_gan)
        ],
        verbose=2
    )

    # ! TSTR goes here (todo)

if __name__ == '__main__':

    # create argparse to get the model name and other parameters
    import argparse
    parser = argparse.ArgumentParser()
    ## & GAN params
    parser.add_argument('--gan_num_epochs', type=int, default=15, help='The number of epochs to train the GAN')
    parser.add_argument('--gan_D_learning_rate', type=float, default=0.0003, help='The learning rate for the discriminator')
    parser.add_argument('--gan_G_learning_rate', type=float, default=0.0003, help='The learning rate for the generator')
    parser.add_argument('--gan_batch_size', type=int, default=32, help='The batch size to use for the GAN')
    parser.add_argument('--gan_latent_dim', type=int, default=128, help='The latent dimension for the GAN')
    parser.add_argument('--use_pretrained_D', action='store_true', help='Use a pretrained discriminator model')
    parser.add_argument('--dataset', type=str, default='processed_dataset', help='The dataset to use', choices=['processed_dataset','processed_dataset_binary'])

    ## & classifcation model params
    parser.add_argument('--classification_model_name', type=str, default='medium', help='The name of the classification to use', choices=MODELS.keys())
    # * add an optional argument for token
    parser.add_argument('--token', type=str, default=None, help='The token to use for the pipeline name')
    args = parser.parse_args()

    # * if the token is provided, use it as the token, otherwise create a new token
    token = args.token if args.token else ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 4))
    print(f"🟢Token: {token}")

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    PIPELINE_NAME = f"classification-{token}"

    run_pipeline(
        PIPELINE_NAME,
        Path(__file__).parent / args.dataset,
        args.gan_num_epochs,
        args.gan_D_learning_rate,
        args.gan_G_learning_rate,
        args.gan_batch_size,
        args.gan_latent_dim,
        args.use_pretrained_D,
        args.classification_model_name,
    )
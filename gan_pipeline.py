# import tf and keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from load_data import load_images
from job_utils import create_output_dir
from gans.cgan import CGAN, GenerateImageCallback, SaveModelCallback
import logging, json

# boilerplate for installing thai font
import matplotlib.font_manager as fm
font_path = str(Path(__file__).parent / 'fonts/NotoSerifThai-Regular.ttf')
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


def run_pipeline(
    model_name: str,
    pipeline_name: str,
    dataset_path: Path,
    gan_num_epochs:int,
    gan_D_learning_rate:float,
    gan_G_learning_rate:float,
    gan_batch_size:int,
    gan_latent_dim:int,

        
):
    output_dir = create_output_dir(pipeline_name, "gan_jobs", skip_if_exists=False)
    output_dir_gan = output_dir / 'gan'
    output_file_name = str((output_dir / f"log.log").resolve())
    # load the data
    train_dataset, validation_dataset, test_dataset, class_names = load_images(dataset_path/'train', dataset_path/'test', gan_batch_size, class_names=class_names)

    # log and print the parameters
    paramaters = {
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

    full_train_dataset = train_dataset.concatenate(validation_dataset)

    # create an instance of the CGAN
    cgan = CGAN(latent_dim=gan_latent_dim, num_classes=len(class_names))

    # compile the model
    cgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=gan_D_learning_rate, beta_1=0.9),
        g_optimizer=keras.optimizers.Adam(learning_rate=gan_G_learning_rate, beta_1=0.9),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=False)
    )    # train the model

    # train the model
    cgan.fit(
        full_train_dataset,
        epochs=gan_num_epochs,
        callbacks=[
            GenerateImageCallback(cgan.generator, gan_latent_dim, len(class_names), output_dir_gan, frequency=1),
            SaveModelCallback(cgan, output_dir_gan)
        ]
    )

    # TSTR goes here (todo)

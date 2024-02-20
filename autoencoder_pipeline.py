import json
from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from load_data import load_images, load_images_for_autoencoder
from job_utils import create_output_dir
import autoencoder.autoencoder_model as autoencoder_model


# boilerplate for installing thai font
font_path = str(Path(__file__).parent / 'fonts/NotoSerifThai-Regular.ttf')
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


def run_autoencoder_pipeline(
        pipeline_name: str,
        dataset_path: Path,
        num_epochs:int,
    ):

    output_dir = create_output_dir(pipeline_name, skip_if_exists=False)
    output_file_name = str((output_dir / f"log.log").resolve())
   
    logging.basicConfig(
        filename=output_file_name,
        encoding="utf-8",
        level=logging.INFO,
        force=True,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # load the data
    train_images, validation_images, test_images = load_images_for_autoencoder(dataset_path/'train', dataset_path/'test')
    # normalise the data in [0, 1]
    train_images = train_images / 255.0
    validation_images = validation_images / 255.0
    test_images = test_images / 255.0

    # create and compile the model
    encoder, autoencoder = autoencoder_model.build_autoencoder_model(128)
    # train the model
    history = autoencoder_model.train_model(autoencoder, train_images, validation_images, model_path=output_dir, epochs=num_epochs)


if __name__ ==  '__main__':
    # create a token consisting of 4 random characters
    token = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 4))
    print(f"🟢Token: {token}")
    PIPELINE_NAME = f"autoencoder-{token}"

    run_autoencoder_pipeline(
        PIPELINE_NAME,
        Path(__file__).parent / 'processed_dataset',
        num_epochs=15
    )

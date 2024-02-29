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
        batch_size:int,
        learning_rate:float,
    ):

    output_dir = create_output_dir(pipeline_name, "autoencoding_jobs", skip_if_exists=False)
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
    # normalise the data in [0, 1] and reverse the colors
    train_images = 1 - (train_images / 255.0)
    validation_images = 1 - (validation_images / 255.0)
    test_images = 1 - (test_images / 255.0)

    # create and compile the model
    encoder, autoencoder = autoencoder_model.build_autoencoder_model(learning_rate=learning_rate)
    # train the model
    history = autoencoder_model.train_model(
        autoencoder,
        train_images,
        validation_images,
        model_path=output_dir,
        epochs=num_epochs,
        batch_size=batch_size
    )
    # save the history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history.history, f)
    # plot the history and save to file
    history_df = pd.DataFrame(history.history)
    history_df.plot()
    plt.savefig(output_dir / 'history.png')
    plt.close()

    # plot the 5 first test images and their reconstructions
    n = 5
    reconstructed_images = autoencoder.predict(test_images[:n])
    fig, axes = plt.subplots(2, n, figsize=(20, 5))
    for i in range(5):
        axes[0, i].imshow((test_images[i]), cmap='gray')
        axes[1, i].imshow((reconstructed_images[i]), cmap='gray')
    plt.savefig(output_dir / 'reconstructed_images.png')
    plt.close()
    # save the model
    autoencoder.save(output_dir / 'model.h5')
    # save the encoder
    encoder.save(output_dir / 'encoder.h5')



if __name__ ==  '__main__':
    # create argparse to get the model name and other parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=15, help='The number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate to use')
    # parser.add_argument('--is_use_augmentation', type=bool, default=True, help='Whether to use data augmentation', action=argparse.BooleanOptionalAction)
    # parser.add_argument('--dataset', type=str, default='processed_dataset', help='The dataset to use', choices=['processed_dataset','processed_dataset_binary'])
    # * add an optional argument for token
    parser.add_argument('--token', type=str, default=None, help='The token to use for the pipeline name')
    args = parser.parse_args()

    PIPELINE_NAME = f"autoencoder-{args.token if args.token else 'default'}"

    run_autoencoder_pipeline(
        PIPELINE_NAME,
        Path(__file__).parent / 'processed_dataset_binary',
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

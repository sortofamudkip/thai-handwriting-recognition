import json
from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd
from load_data import load_images
import classification.classification_model as c_model
import classification.evaluation as c_eval
import classification.visualisation as c_vis
from job_utils import create_output_dir
from classification.classification_model import MODELS

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
        num_epochs:int,
        batch_size:int,
        learning_rate:float,
        is_use_augmentation:bool,
    ):

    output_dir = create_output_dir(pipeline_name, "classification_jobs", skip_if_exists=False)
    output_file_name = str((output_dir / f"log.log").resolve())
   
    logging.basicConfig(
        filename=output_file_name,
        encoding="utf-8",
        level=logging.INFO,
        force=True,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # log and print the parameters
    paramaters = {
        "model_name": model_name,
        "dataset_path": str(dataset_path),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "is_use_augmentation": is_use_augmentation,
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
    train_dataset, validation_dataset, test_dataset, class_names = load_images(dataset_path/'train', dataset_path/'test', batch_size, class_names=class_names)

    # create the model
    model = c_model.get_classification_model(model_name, len(class_names), use_augmentation=is_use_augmentation)

    # train the model
    history = c_model.train_model(model, train_dataset, validation_dataset, model_path=output_dir, epochs=num_epochs,learning_rate=learning_rate)

    # plot the training history and loss history
    c_vis.plot_training_history(history, output_dir / 'training_history.png')
    c_vis.plot_loss_history(history, output_dir / 'loss_history.png')

    # evaluate the model
    y_pred, y_test, raw_predictions = c_eval.get_predictions_and_labels(model, test_dataset)

    # print and log classification report
    report = c_eval.get_classification_report(y_test, y_pred, class_names)
    print(report)
    logging.info(f"Classification report:\n{report}")

    # plot confusion matrix
    confusion = c_eval.get_confusion_matrix(y_test, y_pred)
    c_vis.plot_confusion_matrix(confusion, class_names, output_dir / 'confusion_matrix.png')
    # save the confusion matrix as a json file
    with open(output_dir / 'confusion_matrix.json', 'w') as f:
        json.dump(confusion.tolist(), f)

if __name__ ==  '__main__':
    # create argparse to get the model name and other parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='medium', help='The name of the model to use', choices=MODELS.keys())
    parser.add_argument('--num_epochs', type=int, default=15, help='The number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate to use')
    parser.add_argument('--is_use_augmentation', type=bool, default=True, help='Whether to use data augmentation', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dataset', type=str, default='processed_dataset', help='The dataset to use', choices=['processed_dataset','processed_dataset_binary'])
    # * add an optional argument for token
    parser.add_argument('--token', type=str, default=None, help='The token to use for the pipeline name')
    args = parser.parse_args()

    # * if the token is provided, use it as the token, otherwise create a new token
    token = args.token if args.token else ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 4))
    print(f"🟢Token: {token}")

    PIPELINE_NAME = f"classification-{token}"

    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    run_pipeline(
        model_name=args.model_name,
        pipeline_name=PIPELINE_NAME, 
        dataset_path=Path(__file__).parent / args.dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        is_use_augmentation=args.is_use_augmentation,
    )

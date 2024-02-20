import json
from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from load_data import load_images
import classification.classification_model as c_model
import classification.evaluation as c_eval
import classification.visualisation as c_vis
from job_utils import create_output_dir

# boilerplate for installing thai font
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
        "dataset_path": str(dataset_path),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "is_use_augmentation": is_use_augmentation,
    }
    logging.info(f"Parameters: {paramaters}")
    print(f"Parameters: {paramaters}")


    # load the data
    train_dataset, validation_dataset, test_dataset, class_names = load_images(dataset_path/'train', dataset_path/'test', batch_size)

    # create the model
    model = c_model.get_classification_model(model_name, len(class_names), use_augmentation=is_use_augmentation)

    # train the model
    history = c_model.train_model(model, train_dataset, validation_dataset, model_path=output_dir, epochs=num_epochs,learning_rate=learning_rate)

    # plot the training history and loss history
    c_vis.plot_training_history(history, output_dir / 'training_history.png')
    c_vis.plot_loss_history(history, output_dir / 'loss_history.png')

    # evaluate the model
    y_pred, y_test, raw_predictions = c_eval.get_predictions_and_labels(model, test_dataset)

    # print classification report
    report = c_eval.get_classification_report(y_test, y_pred, class_names)
    print(report)

    # plot confusion matrix
    confusion = c_eval.get_confusion_matrix(y_test, y_pred)
    c_vis.plot_confusion_matrix(confusion, class_names, output_dir / 'confusion_matrix.png')
    # save the confusion matrix as a json file
    with open(output_dir / 'confusion_matrix.json', 'w') as f:
        json.dump(confusion.tolist(), f)

if __name__ ==  '__main__':
    # create a token consisting of 4 random characters
    token = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 4))
    print(f"🟢Token: {token}")
    PIPELINE_NAME = f"classification-{token}"

    run_pipeline(
        model_name='medium',
        pipeline_name=PIPELINE_NAME, 
        dataset_path=Path(__file__).parent / 'processed_dataset',
        num_epochs=15,
        batch_size=32,
        learning_rate=0.001,
        is_use_augmentation=False,
    )

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

def create_output_dir(pipeline_name: str, skip_if_exists=False) -> Path:
    """
    Creates a new directory for the pipeline's output files.

    Args:
        pipeline_name (str): The name of the pipeline.
        skip_if_exists (bool): If True, returns the existing directory instead of creating a new one.

    Returns:
        Path: The path to the newly created directory.
    """
    output_dir = (Path(__file__).parent / "./jobs/results" / pipeline_name).resolve()
    if skip_if_exists and output_dir.is_dir():
        return output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        logging.info(f"Created output dir: {output_dir}")
    except FileExistsError:
        logging.error(f"Dir already exists: {output_dir}")
        assert False
    return output_dir

def run_pipeline(
        pipeline_name: str,
        dataset_path: Path,
        num_epochs:int,
    ):
    # boilerplate for installing thai font
    font_path = 'Ayuthaya.ttf'
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)

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
    train_dataset, validation_dataset, test_dataset, class_names = load_images(dataset_path/'train', dataset_path/'test', 32)

    # create the model
    model = c_model.create_classification_model((64, 64, 1), len(class_names), is_use_augmentation=False)

    # train the model
    history = c_model.train_model(model, train_dataset, validation_dataset, model_path=output_dir, epochs=num_epochs)

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

    run_pipeline(PIPELINE_NAME, Path(__file__).parent / 'processed_dataset', num_epochs=5)

from load_data import load_images
from classification.tuning import build_hp_model
from tensorflow import keras
import keras_tuner as kt

import matplotlib.pyplot as plt
from pathlib import Path
# boilerplate for installing thai font
import matplotlib.font_manager as fm
font_path = str(Path(__file__).parent / 'fonts/NotoSerifThai-Regular.ttf')
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

def build_tuner(is_distribution_strategy=False):
    random_token = ''.join([str(i) for i in range(10)])
    tuner = kt.Hyperband(
        build_hp_model, # * model building function
        objective='val_sparse_categorical_accuracy', 
        max_epochs=15, # * max epochs to train the model in each trial
        factor=3, # * factor that determines the number of models to train in each bracket
        hyperband_iterations=1, # * number of times to iterate over the full hyperband algorithm
        directory=f'tuner_{random_token}', # * directory to save the results
        project_name=f'classification_{random_token}', # * project name,
        distribution_strategy= is_distribution_strategy if is_distribution_strategy else None,
    )
    return tuner


if __name__ == "__main__":

    BATCH_SIZE = 32
    class_names = 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
    # class_names = 'กฮ' # ! test

    # load the data
    dataset_path = Path(__file__).parent / "processed_dataset"
    # dataset_path = Path(__file__).parent / "processed_dataset_binary" # ! test
    train_dataset, validation_dataset, test_dataset, class_names = load_images(dataset_path/'train', dataset_path/'test', BATCH_SIZE, class_names=class_names)

    # * define early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # * build the tuner
    tuner = build_tuner(None)
    tuner.search(train_dataset, validation_data=validation_dataset, epochs=15, verbose=2, callbacks=[early_stopping])
    tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    best_model.summary()
    
    # * train the best model
    best_model.fit(train_dataset, validation_data=validation_dataset, epochs=100, verbose=2)

    # * try saving the best model
    try:
        best_model.save('best_model.h5')
    except Exception as e:
        print("Error saving the best model")
        print(e)
    # * try saving the best model weights
    try:
        best_model.save_weights('best_model_weights.h5')
    except Exception as e:
        print("Error saving the best model weights")
        print(e)
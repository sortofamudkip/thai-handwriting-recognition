# thai-handwriting-recognition

My code for the final project of the IDA-ML2 class.

## Overview

**The code most relevant for the presentation is in `classification/` and `classification_pipeline.py`.**

Though there is a lot of code relating to different parts of the course, such as GANs and autoencoders, I did not find the results to these parts satisfactory enough to include them in my presentation. Other auxilliary files, such as `load_data.py`, are also explained below. 

## Relevant files
### `classification_pipeline.py`
A script that runs the entire classification pipeline using code from `classification/`, in particular data loading, model instantiation and training, and finally evaluation and saving results.

### `classification/`
The source code for the classification task.

### `classification/cnn_models/`
Where the models are defined. The models relevant to the presentation are:
* `basic.py`, the baseline model
* `medium_model.py`, the specilised model
* `mobilenet.py`, the transfer learning model

Other files are less developed and not included in the presentation.

### `classification/classification_model.py`
The code for selecting and instantiating the model. The data augmentation layers and model training callbacks are also defined here.

### `classification/evaluation.py`
The code for obtaining classification metrics, namely the accuracy and F1 score.

### `classification/tuning.py`
the code for tuning the (specialised) model is stored here.

### `classification/visualisation.py`
This file contains the code used for visualising the learning curves and confusion matrices.

### `classification/embedding_and_PCA.py`
The code for image embeddings and PCA visualisation was initially written in a Jupyter Notebook. They are kept in this file for reference but not used in the pipeline.

### `load_data.py`
Code that loads the images.

## Miscellaneous
The two contributors are both me, pushing code from either my local laptop or the HPC.

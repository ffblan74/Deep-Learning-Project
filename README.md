# Deep Learning Project: Multi-Modal Classification (Vision & NLP)

This project implements a multi-modal fusion architecture combining:

- A Convolutional Neural Network (CNN) for computer vision, and
- A Recurrent Neural Network (RNN) for natural language processing.

The objective is to perform classification or analysis based on the Flickr8k dataset.

---

## Prerequisites and Data

The project requires the following data, which must be downloaded and extracted at the root of the project:

- Flickr8k Token (Text):
  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

- Flickr8k Dataset (Images):
  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip

- GloVe Embeddings:
  https://nlp.stanford.edu/data/glove.6B.zip
  
  Use the file `glove.6B.300d.txt`.

---

## Configuration

Before running the project, ensure that the paths to the downloaded files are correctly configured in the `main.py` and `test.py` files.

You can modify these variables at the beginning of the scripts to adapt them to your directory structure:

### Global Configuration (main.py and test.py)

```python
IMAGE_DIR = "Flicker8k_Dataset"      #  # Folder containing the images (note: located inside the "Flickr8k_Dataset" directory")
TOKEN_FILE = "Flickr8k.token.txt"    # File containing the descriptions
GLOVE_FILE = "glove.6B.300d.txt"     # GloVe word vectors
```

---

## Project Structure

The model input consists of:

- A raw image, processed by the vision branch (Part 1), and
- Its associated textual description, processed by the text branch (Part 2).

The two representations are then fused to produce the final prediction.

---

## Modules Folder

### vision_part1.py (Part 1)

This module is responsible for image processing.

It uses a pre-trained convolutional neural network (ResNet50) to extract visual features or to load raw image pixels required for Fine-Tuning.

Each image is transformed into a numerical representation usable by the fusion model.

---

### text_part2.py (Part 2)

This script handles the textual component of the project.

It cleans the image descriptions, performs tokenization, and prepares the embedding matrix using GloVe word vectors.

It uses a recurrent neural network (LSTM or GRU) to convert word sequences into semantic vectors that capture contextual information.

---

### fusion_part3.py (Part 3)

This file defines the global multi-modal fusion architecture.

It retrieves the output vectors from the vision and text modules, concatenates them, and passes the fused representation through fully connected (Dense) layers to produce the final prediction.

---

## Main Files

### main.py

This script orchestrates data loading and the complete training process.

It executes the training loop, manages Cross-Validation to tune hyperparameters such as Dropout, and saves the final optimized model into an `.h5` file.

---

### test.py

This script is used for inference.

It allows testing the trained model on a random image to verify its performance.

---

## Usage

### Quick Test (Pre-trained Model)

A trained model (`modele_fusion.h5`) is already provided with this project. You can directly run the test script without retraining the model:

```bash
python test.py
```

---

### Full Training

If you wish to retrain the model from scratch, including Fine-Tuning, run:

```bash
python main.py
```

Note: Full training is resource-intensive and requires a suitable configuration with sufficient RAM or a GPU.

---

## Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---
## Authors

- Hugo LEFEVRE  
- Camille CHAPTINI  
- Félix BLANCHIER

Project carried out as part of the Deep Learning course - Year 2025/2026


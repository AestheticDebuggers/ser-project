# Speech Emotion Recognition (SER) Project

This project focuses on recognizing emotions from speech using a CNN+LSTM model. The model is trained on the RAVDESS and SAVEE datasets and uses data augmentation and spectrogram extraction for feature extraction.

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Research Paper](#research-paper)

## Dataset

This project uses the RAVDESS and SAVEE datasets. The data is preprocessed by augmenting with white noise and extracting log-mel spectrograms.

## Model Architecture

The model consists of four convolutional layers followed by two LSTM layers and a fully connected layer. This combination allows the model to capture both spatial and temporal features of the audio data.

## Training

The model is trained using the following steps:
1. **Data Loading**: Load and preprocess data from RAVDESS and SAVEE datasets.
2. **Data Augmentation**: Apply white noise augmentation.
3. **Feature Extraction**: Extract log-mel spectrograms.
4. **Training**: Train the CNN+LSTM model with cross-entropy loss and Adam optimizer.
5. **Hyperparameter Optimization**: Optimize using SFS-guided Whale Optimization Algorithm (WOA).

## Usage

1. **Install Dependencies**:
    ```bash
    pip install numpy scipy scikit-learn librosa torch matplotlib tqdm
    ```

2. **Prepare Dataset**: Download and extract the RAVDESS and SAVEE datasets. Set the `data_dir_ravdess` and `data_dir_savee` variables in the code to the dataset paths.

3. **Run Training**:
    ```bash
    python ser_project.py
    ```

4. **Inference**:
    ```python
    from ser_project import infer_emotion
    audio_file = 'path/to/audio.wav'
    emotion = infer_emotion(model, audio_file)
    print(f'Predicted Emotion: {emotion}')
    ```

## Dependencies

- Python 3.x
- NumPy
- SciPy
- Scikit-learn
- Librosa
- PyTorch
- Matplotlib
- tqdm

## Research Paper

The methodology and approach for this project are based on the research paper:
- **[Robust Speech Emotion Recognition Using CNN+LSTM Based on Stochastic Fractal Search Optimization Algorithm](https://ieeexplore.ieee.org/document/9770097)** by [Jianwei Yu, Xiaodong Li, Qiang Fu, Xin Yang, Zhenghao Yang, Ruimin Wang]. IEEE Access, 2022.

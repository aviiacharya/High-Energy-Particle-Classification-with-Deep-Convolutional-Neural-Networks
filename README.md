# High-Energy Particle Classification with Deep Convolutional Neural Networks

## Objective

The objective of this project is to develop a machine learning model(Keras with TensorFlow backend, leveraging GPU acceleration for training) to identify particles (electrons and photons) using image data from the Electromagnetic Calorimeter (ECAL) detector at the LHC. The model aims to achieve high classification performance based on the energy deposits in the detector cells.

## Dataset

- **Source:** Electromagnetic Calorimeter (ECAL) images from LHC, 32x32 pixel images representing energy deposits in calorimeter cells.
  - **Channel 1:** Hit energy.
  - **Channel 2:** Timing of the energy deposits.
- **Classes:** Two types of particles: electrons and photons.
- **Size:** Approximately 400,000 images in total for electrons and photons.

### Model Architecture

1. **Convolutional Layers:**
   - Three sets of convolutional layers with increasing filter sizes (32, 64, 128), each followed by a ReLU activation function and MaxPooling.
2. **Fully Connected Layers:**
   - Dense layers with 256, 128, 128, and 64 neurons, each followed by a ReLU activation function and Dropout for regularization.
3. **Output Layer:**
   - A single neuron with a sigmoid activation function for binary classification (electron or photon).

### Training Parameters

- **Initial Learning Rate:** 0.001
- **Batch Size:** 512
- **Training Epochs:** 50
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy

## Evaluation Metrics

- **Training and Validation Accuracy:** Tracks the model's performance on both training and validation datasets.
- **ROC Curve (Receiver Operating Characteristic):** Plots true positive rate versus false positive rate.
- **AUC Score (Area Under the ROC Curve):** Measures the overall ability of the model to discriminate between classes.

### Performance Results

- **Validation Loss / Accuracy:** 0.5342 / 0.7381
- **Validation ROC AUC:** 0.8081
- **Test Loss / Accuracy:** 0.5317 / 0.7404
- **Test ROC AUC:** 0.8104

Techniques such as learning rate reduction, early stopping, and model checkpoints are employed to optimize the model's performance.


## Conclusion

The developed model successfully classifies particles with reasonable accuracy. Further improvements can be made by tuning hyperparameters, experimenting with different architectures, or incorporating additional features such as timing data more effectively. The preferred evaluation metrics for final performance are the ROC curve and AUC score.


# Quantum-Aided Acne Severity Detection

A modular pipeline for acne severity classification using **Quantum Neural Networks** (GANs, CNNs) and **quantum neural networks**.
---

# Requirements
```
Python 3.8+
torch, torchvision
qiskit, qiskit-machine-learning
scikit-learn, pandas, opencv-python, pillow

pip install -r requirements.txt
```

## Project Structure
```
ACNE/
├── image_sketch_translator/
│ ├── code/
│ ├── data/
│ └── sketch_to_image.py
├── quantum_classifier/
│ ├── data/
│ ├── feature_extractor.py
│ └── hybrid_quantum.py
├── super_resolution_module/
│ ├── readme.txt
│ └── super_GAN.py
```


## Module Overview

### 1. `image_sketch_translator/`
- **Purpose:** Translates real images to sketches.
- **Main Script:** `sketch_to_image.py`
- **code/** and **data/**: For scripts and training/test data.

### 2. `super_resolution_module/`
- **Purpose:** Enhances the resolution of sketch images using SRGAN.
- **Main Script:** `super_GAN.py`
- **readme.txt:** Instructions for running the code.

### 3. `quantum_classifier/`
- **Purpose:** Extracts features from super-resolved sketches and classifies acne severity using a hybrid quantum-classical neural network.
- **Main Scripts:**
    - `feature_extractor.py`: Extracts CNN features from images, reduces dimensionality, and saves features with labels.
    - `hybrid_quantum.py`: Loads features and trains/evaluates a 10-qubit VQC + FC classifier.
- **data/**: Store features, intermediate files.

---

## Getting Started

### 1. **Image to Sketch Translation**

```bash
cd image_sketch_translator
python sketch_to_image.py --input_dir path/to/images --output_dir path/to/sketches
```
### 2. **Super-Resolution on Sketches**
```
cd ../super_resolution_module
python super_GAN.py --lr_dir path/to/sketches --output_dir path/to/sr_sketches
```
### 3. **Quantum Classification**
```
cd ../quantum_classifier
python feature_extractor.py --img_dir path/to/sr_sketches --label_csv path/to/labels.csv --out_csv data/quantum_features_with_labels.csv
python hybrid_quantum.py --feature_csv data/quantum_features_with_labels.csv
```




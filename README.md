# 🌿 Plant Disease Intelligence

### Deep Learning • Computer Vision • Transfer Learning • Explainable AI • Streamlit

An end-to-end Deep Learning system for **plant disease classification** using **EfficientNetB0 transfer learning**, selective fine-tuning, and **Grad-CAM explainability**.

The system classifies plant leaf images across **281 disease classes** and provides:

- 🌱 Predicted plant disease
- 📊 Prediction confidence
- 🔎 Top-5 alternative predictions
- 🧠 Grad-CAM visual explanation
- 📈 Model performance metrics
- 🖥️ Interactive Streamlit application

The project covers the complete workflow from **dataset exploration and preprocessing to model training, evaluation, explainable AI, application development, and deployment**.

---

# 🚀 Live Application

🔗 **👉 Open Plant Disease Intelligence App**

> Deployment URL will be added after Streamlit Community Cloud deployment.

The application allows users to upload a plant leaf image and receive a disease classification result through an interactive web interface.

---

# 📌 Project Overview

Plant diseases can significantly affect agricultural productivity and crop quality.

Many plant diseases produce visible symptoms such as:

- Leaf spots
- Lesions
- Mosaic patterns
- Yellowing
- Blight
- Mildew
- Discoloration
- Necrosis
- Texture changes

This project explores how **Deep Learning and Computer Vision** can be used to automatically classify plant leaf images into multiple disease categories.

The developed system uses **EfficientNetB0**, a convolutional neural network pretrained on ImageNet, and adapts it to a fine-grained **281-class plant disease classification problem**.

The project also integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the image regions that contributed to the model's prediction.

---

# 🎯 Objectives

The main objectives of this project are:

- Build a multi-class plant disease classification system.
- Work with a large-scale image dataset containing approximately 66,953 images.
- Analyze plant species and disease-class distributions.
- Explore image dimensions and dataset splits.
- Validate disease class IDs and class mappings.
- Build an efficient TensorFlow `tf.data` input pipeline.
- Standardize input images to `224 × 224 × 3`.
- Apply image augmentation to the training dataset.
- Implement transfer learning using ImageNet-pretrained EfficientNetB0.
- Build a custom 281-class classification head.
- Train the model using a two-stage transfer learning strategy.
- Fine-tune deeper EfficientNetB0 layers using a lower learning rate.
- Evaluate the model using Top-1, Top-5, Precision, Recall, and F1-score.
- Analyze model predictions and confidence scores.
- Implement Grad-CAM for visual explainability.
- Save the trained model and supporting metadata.
- Build an interactive Streamlit application.
- Prepare the project for GitHub and cloud deployment.

---

# 🧠 End-to-End Deep Learning Workflow

The project follows the complete Deep Learning lifecycle:

```text
Master Plant Disease Dataset
             ↓
       Dataset Loading
             ↓
  Exploratory Data Analysis
             ↓
    Class ID Validation
             ↓
    Train / Validation / Test
             ↓
      Image Preprocessing
             ↓
        Image Resize
        224 × 224 × 3
             ↓
      Data Augmentation
             ↓
       tf.data Pipeline
             ↓
      EfficientNetB0
     ImageNet Pretrained
             ↓
     Global Average Pooling
             ↓
      Batch Normalization
             ↓
        Dropout (0.4)
             ↓
      Dense (281 Units)
             ↓
        Softmax Output
             ↓
      Initial Training
             ↓
       Fine-Tuning
             ↓
       Model Evaluation
             ↓
        Grad-CAM
             ↓
       Model Serialization
             ↓
      Streamlit Application
             ↓
        Cloud Deployment
```

---

# 📊 Dataset

The project uses the **Master Plant Disease Dataset** sourced from Kaggle.

**Dataset:**  
https://www.kaggle.com/datasets/harisri2005/plant-disease-processed

## Dataset Specifications

| Property | Specification |
|---|---|
| Dataset | Master Plant Disease Dataset |
| Total Images | ~66,953 |
| Number of Classes | 281 |
| Class IDs | 0 – 280 |
| Plant Species | 42 |
| Dataset Size | ~4.48 GB |
| Image Type | RGB |
| Model Input | 224 × 224 × 3 |
| Dataset Splits | Training / Validation / Test |

The dataset contains images representing multiple agricultural and horticultural plant species and disease categories.

---

# 📁 Dataset Organization

The dataset contains the following major components:

```text
Dataset/
│
├── master_images/
│   └── ~66,953 plant leaf images
│
├── metadata/
│   ├── source_manifest_full.csv
│   ├── class_id_map.csv
│   ├── class_map.csv
│   ├── dataset_manifest.csv
│   └── dedup_manifest.csv
│
└── outputs/
    ├── train_split.csv
    ├── val_split.csv
    └── test_split.csv
```

### `master_images/`

Contains the original plant leaf images used during model development and evaluation.

### `metadata/`

Contains dataset metadata, class mappings, source information, and manifest files.

### `outputs/`

Contains the predefined training, validation, and testing split information.

---

# 🔍 Exploratory Data Analysis

Before model development, Exploratory Data Analysis was performed to understand the structure and distribution of the dataset.

The analysis included:

### 🌱 Plant Species Analysis

The dataset was analyzed to identify the different plant species represented in the dataset.

### 🦠 Disease Class Analysis

The disease classes were examined to understand the number and distribution of disease categories.

The dataset contains:

```text
281 disease classes
Class IDs: 0 → 280
```

### 📊 Source Dataset Distribution

The distribution of images from different source datasets was analyzed to understand the composition of the overall dataset.

### 📐 Image Dimension Analysis

The original images contain varying native resolutions.

Image dimensions were analyzed before preprocessing to understand the variability of the source images.

Because EfficientNetB0 requires a consistent input shape, all images were standardized to:

```text
224 × 224 × 3
```

### 🖼️ Random Image Visualization

Random image samples were visualized to inspect:

- Plant leaves
- Disease symptoms
- Image quality
- Background variation
- Lighting conditions
- Visual differences between disease categories

### 📦 Dataset Split Analysis

The number of images in the following subsets was analyzed:

- Training set
- Validation set
- Test set

---

# 🧾 Class ID Validation

Before model construction, the class IDs were validated.

The dataset was confirmed to contain:

```text
Minimum Class ID: 0
Maximum Class ID: 280
Number of Classes: 281
Contiguous IDs: True
```

This validation was important because the neural network output layer directly corresponds to the class IDs.

The final classification layer therefore contains:

```text
281 output neurons
```

with a Softmax activation.

---

# 🛠️ Data Preprocessing

The raw images contain different resolutions and visual characteristics.

A standardized preprocessing pipeline was therefore created.

## 1. Image Resizing

All images are resized to:

```text
224 × 224 × 3
```

The `224 × 224` input size was selected to match the EfficientNetB0 model configuration used in the project.

---

## 2. RGB Conversion

Images are decoded with three color channels:

```text
RGB
```

This ensures compatibility with the pretrained EfficientNetB0 architecture.

---

## 3. Data Type Conversion

Images are converted to:

```text
float32
```

for TensorFlow model processing.

The preprocessing pipeline keeps the image values compatible with the TensorFlow/Keras EfficientNet implementation.

---

# 🔄 TensorFlow Data Pipeline

The input pipeline was implemented using:

```python
tf.data.Dataset
```

The pipeline performs:

```text
Image Path
    ↓
Read Image
    ↓
Decode Image
    ↓
Resize
    ↓
Convert to float32
    ↓
Batch
    ↓
Prefetch
```

The pipeline uses:

```python
tf.data.AUTOTUNE
```

for optimized data prefetching and input processing.

---

# 📦 Batch Size

The training pipeline uses:

```text
Batch Size = 16
```

The batch size was selected based on the available computing resources and memory requirements of the EfficientNetB0 training process.

---

# 🎨 Data Augmentation

Data augmentation was applied only to the training dataset.

The augmentation pipeline includes:

```python
RandomFlip("horizontal")
RandomRotation(0.1)
RandomZoom(0.1)
RandomContrast(0.1)
```

The purpose of augmentation is to expose the model to realistic variations in plant leaf images.

These transformations can help the model become less dependent on:

- Image orientation
- Minor rotations
- Zoom level
- Contrast
- Lighting variation

Validation and test datasets remain unaugmented so that model performance is evaluated without applying training-time transformations.

---

# 🧠 Deep Learning Model

The project uses:

## EfficientNetB0

EfficientNetB0 was selected as the main convolutional neural network backbone.

The model uses:

```text
EfficientNetB0
Pretrained on ImageNet
```

Transfer learning allows the project to start with visual features learned from a large image dataset instead of training an entire convolutional neural network from scratch.

---

# 🏗️ Model Architecture

The final architecture is:

```text
Input Image
224 × 224 × 3
      │
      ▼
EfficientNetB0
ImageNet Pretrained
      │
      ▼
GlobalAveragePooling2D
      │
      ▼
BatchNormalization
      │
      ▼
Dropout
0.4
      │
      ▼
Dense Layer
281 Units
      │
      ▼
Softmax
      │
      ▼
281 Disease Probabilities
```

---

# 🔬 Model Components

## 1. EfficientNetB0

EfficientNetB0 acts as the main feature extractor.

It learns visual representations such as:

- Edges
- Textures
- Shapes
- Color patterns
- Leaf structures
- Disease-related visual patterns

---

## 2. GlobalAveragePooling2D

Global Average Pooling converts the spatial feature maps into a compact feature representation.

This reduces the number of trainable parameters compared with using a large fully connected layer directly on the feature maps.

---

## 3. Batch Normalization

Batch Normalization is applied after Global Average Pooling.

It helps stabilize the feature representation during training.

---

## 4. Dropout

The classification head uses:

```text
Dropout = 0.4
```

Dropout helps reduce overfitting by randomly disabling a portion of the classification neurons during training.

---

## 5. Dense Classification Layer

The final Dense layer contains:

```text
281 neurons
```

because the dataset contains 281 disease classes.

The activation function is:

```text
Softmax
```

Softmax converts the model output into a probability distribution across all 281 disease classes.

---

# 🔄 Transfer Learning Strategy

The project uses a two-stage training strategy.

## Phase 1 — Feature Extraction

Initially, the EfficientNetB0 backbone was frozen.

```python
base_model.trainable = False
```

Only the newly added classification head was trained.

The optimizer used:

```text
Adam
Learning Rate = 1 × 10⁻³
```

The loss function was:

```text
Sparse Categorical Crossentropy
```

This phase allows the newly added classification head to learn how to map EfficientNetB0 features to the 281 plant disease categories.

---

# 🔬 Phase 2 — Fine-Tuning

After the initial training phase, deeper layers of EfficientNetB0 were selectively unfrozen.

Earlier layers remained frozen while later layers were allowed to adapt to plant disease-specific visual features.

A much smaller learning rate was used:

```text
Learning Rate = 1 × 10⁻⁵
```

Using a lower learning rate helps preserve useful pretrained features while allowing the model to adapt to the plant disease domain.

Fine-tuning allows the model to better learn:

- Leaf textures
- Lesion boundaries
- Disease spots
- Discoloration
- Mosaic patterns
- Plant-specific visual characteristics

---

# ⚙️ Training Configuration

| Parameter | Configuration |
|---|---|
| Backbone | EfficientNetB0 |
| Pretrained Weights | ImageNet |
| Input Shape | 224 × 224 × 3 |
| Number of Classes | 281 |
| Batch Size | 16 |
| Optimizer | Adam |
| Phase 1 Learning Rate | 1e-3 |
| Phase 2 Learning Rate | 1e-5 |
| Loss | Sparse Categorical Crossentropy |
| Dropout | 0.4 |
| Initial Training | Up to 10 epochs |
| Fine-Tuning | Up to 10 epochs |

---

# ⏱️ Training Callbacks

Several callbacks were used to improve training stability.

## EarlyStopping

Training monitored validation loss and restored the best model weights.

This helps prevent unnecessary training after validation performance stops improving.

---

## ReduceLROnPlateau

The learning rate was reduced when validation loss stopped improving.

Configuration:

```text
Factor = 0.2
Patience = 2
Minimum Learning Rate = 1e-7
```

---

## ModelCheckpoint

The best-performing model was saved using:

```text
models/plant_disease_best.keras
```

This saved model is used by the Streamlit application for inference.

---

# 💾 Saved Model Artifacts

The trained models and supporting configuration files are stored inside:

```text
models/
```

The main files are:

| File | Purpose |
|---|---|
| `plant_disease_best.keras` | Best trained model used for deployment |
| `plant_disease_efficientnetb0.keras` | Saved EfficientNetB0 model |
| `class_names.json` | Class ID to disease-name mapping |
| `config.json` | Model and preprocessing configuration |
| `model_metadata.json` | Model performance and metadata |

---

# 📈 Model Evaluation

The trained model was evaluated on the held-out test dataset.

The project evaluates both overall classification performance and class-balanced performance.

The evaluation includes:

- Accuracy
- Top-5 Accuracy
- Macro Precision
- Macro Recall
- Macro F1-Score
- Confusion Matrix
- Prediction Confidence

---

# 🏆 Final Model Performance

The current model evaluation results are:

| Metric | Score | Description |
|---|---:|---|
| Test Accuracy / Top-1 | **81.49%** | Correct class ranked first |
| Top-5 Accuracy | **94.20%** | Correct class appears within the top five predictions |
| Macro Precision | **44.65%** | Average precision across all 281 classes |
| Macro Recall | **42.74%** | Average recall across all 281 classes |
| Macro F1-Score | **42.28%** | Class-balanced harmonic mean of precision and recall |

The Top-5 metric is particularly useful for a 281-class fine-grained classification problem because visually similar diseases can produce competing predictions.

---

# 📊 Evaluation Metrics Explained

## Accuracy

Accuracy measures the percentage of test images for which the highest-probability prediction matches the actual class.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

## Top-5 Accuracy

Top-5 Accuracy measures whether the correct disease appears among the model's five highest-probability predictions.

This is useful when several disease classes have visually similar symptoms.

---

## Macro Precision

Precision is calculated independently for each class and then averaged.

Macro averaging gives every disease class equal importance.

---

## Macro Recall

Recall measures how effectively the model identifies examples belonging to each disease class.

Macro Recall provides a class-balanced view instead of allowing frequently occurring classes to dominate the metric.

---

## Macro F1-Score

Macro F1 combines precision and recall for each class and then averages the values equally across all classes.

It provides a class-balanced measure of classification performance.

---

# 🔎 Confusion Matrix Analysis

A confusion matrix was generated during model evaluation to understand class-level prediction behavior.

The normalized confusion matrix helps identify:

- Frequently confused disease classes
- Classes with strong recognition
- Classes with weak recall
- Visually similar disease categories
- Potential effects of class distribution

This analysis provides additional information that cannot be obtained from overall accuracy alone.

---

# 📊 Prediction Confidence Analysis

The model produces a probability distribution across all 281 disease classes.

For each uploaded image:

```text
Input Image
     ↓
281 Class Probabilities
     ↓
Highest Probability
     ↓
Top-1 Prediction
```

The application also displays the five highest-probability classes.

Example:

```text
1. Blackgram - Yellow Mosaic      94.41%
2. Groundnut - Late leaf spot      1.40%
3. Tomato - Leaf Mold              0.63%
4. Tomato - tomato early blight    0.58%
5. Apple - Scab                    0.41%
```

The Top-5 distribution allows users to see alternative model predictions instead of only the highest-probability class.

---

# 🧠 Explainable AI — Grad-CAM

A major component of this project is **Grad-CAM**.

Grad-CAM stands for:

```text
Gradient-weighted Class Activation Mapping
```

It provides a visual explanation of the image regions that contributed to a model prediction.

---

# 🔬 How Grad-CAM Works

The Grad-CAM pipeline follows:

```text
Input Leaf Image
       ↓
EfficientNetB0
       ↓
Final Convolutional Feature Maps
       ↓
Predicted Disease
       ↓
Gradient Calculation
       ↓
Gradient-weighted Feature Maps
       ↓
Activation Heatmap
       ↓
Resize Heatmap
       ↓
Heatmap Overlay
```

The implementation uses the final convolutional activation layer:

```text
top_activation
```

from the EfficientNetB0 backbone.

---

# 🌡️ Grad-CAM Visualization

The Streamlit application displays:

```text
Original Leaf Image
        +
Grad-CAM Heatmap
        ↓
Visual Explanation
```

The heatmap highlights image regions that contributed strongly to the selected prediction.

Potentially highlighted areas can include:

- Leaf lesions
- Spots
- Discoloration
- Infected margins
- Mosaic patterns
- Other visually distinctive regions

Grad-CAM is used as an interpretability tool to inspect which visual regions influenced the prediction.

---

# 🖥️ Streamlit Application

The trained model was integrated into an interactive Streamlit application.

The application is designed for **inference only**.

Users do not need to run the training notebook to use the deployed application.

---

# 🚀 Streamlit Features

## 📤 Image Upload

Users can upload plant leaf images in:

```text
JPG
JPEG
PNG
```

formats.

---

## 🌱 Disease Prediction

After uploading an image, the application displays:

- Predicted disease
- Confidence score
- Top-5 predictions

---

## 📊 Top-5 Prediction Distribution

The application displays the five highest-probability disease classes.

This provides additional information when multiple classes have similar visual characteristics.

---

## 🧠 Grad-CAM Explanation

The application generates a Grad-CAM visualization for the prediction.

Users can compare:

```text
Original Image
```

with:

```text
Grad-CAM Visualization
```

---

## 📈 Model Performance

The application reads saved model metadata and displays:

- Test Accuracy
- Top-5 Accuracy
- Macro Precision
- Macro Recall
- Macro F1

---

# 🔄 Streamlit Prediction Pipeline

The deployed application follows this inference workflow:

```text
User Uploads Leaf Image
          ↓
PIL Image
          ↓
RGB Conversion
          ↓
Resize to 224 × 224
          ↓
Convert to float32
          ↓
Saved EfficientNetB0 Model
          ↓
281-Class Softmax Prediction
          ↓
Top-1 Prediction
          ↓
Top-5 Predictions
          ↓
Confidence Visualization
          ↓
Grad-CAM Explanation
```

---

# 🧩 Deployment Architecture

The project separates model development from application inference.

```text
                    GitHub Repository
                           │
          ┌────────────────┴────────────────┐
          │                                 │
    Jupyter Notebook                    Saved Model
          │                                 │
          │                         plant_disease_best.keras
          │                                 │
          │                           class_names.json
          │                                 │
          │                           model_metadata.json
          │                                 │
          └────────────────┬────────────────┘
                           ↓
                    Streamlit Application
                           ↓
                 Streamlit Community Cloud
                           ↓
                     Public Web App
```

The original 66,953-image dataset is required for model development and training but is not required for inference after the trained model has been saved.

---

# 📁 Project Structure

```text
Plant Disease Classification-DEEP LEARNING/
│
├── Dataset/
│   ├── master_images/
│   │   └── ~66,953 plant images
│   │
│   ├── metadata/
│   │   ├── source_manifest_full.csv
│   │   ├── class_id_map.csv
│   │   ├── class_map.csv
│   │   ├── dataset_manifest.csv
│   │   └── dedup_manifest.csv
│   │
│   └── outputs/
│       ├── train_split.csv
│       ├── val_split.csv
│       └── test_split.csv
│
├── models/
│   ├── class_names.json
│   ├── config.json
│   ├── model_metadata.json
│   ├── plant_disease_best.keras
│   └── plant_disease_efficientnetb0.keras
│
├── streamlit_app/
│   ├── app.py
│   └── requirements.txt
│
├── .streamlit/
│   └── config.toml
│
├── .gitignore
│
├── README.md
│
└── Plant Disease Classification-notebook.ipynb
```

---

# 📓 Jupyter Notebook

The main development notebook is:

```text
Plant Disease Classification-notebook.ipynb
```

The notebook contains the model development workflow, including:

```text
Dataset Loading
       ↓
Dataset Exploration
       ↓
EDA
       ↓
Class Validation
       ↓
Preprocessing
       ↓
TensorFlow Data Pipeline
       ↓
Data Augmentation
       ↓
EfficientNetB0
       ↓
Transfer Learning
       ↓
Fine-Tuning
       ↓
Evaluation
       ↓
Confusion Matrix
       ↓
Grad-CAM
       ↓
Model Saving
```

The notebook serves as the main experimentation and training environment, while Streamlit serves as the final inference interface.

---

# 🛠️ Technologies Used

## Programming

- Python

## Deep Learning

- TensorFlow
- Keras
- EfficientNetB0
- Transfer Learning
- Fine-Tuning

## Data Processing

- NumPy
- Pandas
- Pillow

## Machine Learning Evaluation

- Scikit-learn

## Visualization

- Matplotlib

## Explainable AI

- Grad-CAM
- TensorFlow GradientTape

## Application Development

- Streamlit

## Development Environment

- Jupyter Notebook
- Anaconda / Python Environment

## Version Control

- Git
- GitHub

## Deployment

- Streamlit Community Cloud

---

# 📦 Python Libraries

The main libraries used by the project include:

```text
tensorflow
numpy
pandas
scikit-learn
pillow
matplotlib
streamlit
```

The Streamlit application has its own dependency file:

```text
streamlit_app/requirements.txt
```

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project directory:

```bash
cd "Plant Disease Classification-DEEP LEARNING"
```

---

# 🐍 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

# 📦 3. Install Dependencies

Install the Streamlit application dependencies:

```bash
pip install -r streamlit_app/requirements.txt
```

---

# ▶️ Run the Application Locally

Run the application from the **project root directory**:

```bash
streamlit run streamlit_app/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 🧪 Running the Project for Model Development

The training workflow is available in:

```text
Plant Disease Classification-notebook.ipynb
```

The notebook can be opened using Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

The dataset should be available inside:

```text
Dataset/
```

before running the training workflow.

---

# ☁️ Streamlit Community Cloud Deployment

The application can be deployed using Streamlit Community Cloud.

The deployment architecture is:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
streamlit_app/app.py
       ↓
streamlit_app/requirements.txt
       ↓
models/
       ↓
EfficientNetB0 Model
       ↓
Public Streamlit Application
```

The Streamlit application uses project-relative paths so that the application can work locally and during cloud deployment without changing the model locations.

---

# 🌐 Using the Application on Other Devices

After deployment, the application receives a public Streamlit URL.

The URL can be opened from:

- Laptop
- Desktop
- Tablet
- Smartphone

Users only need a web browser.

They do not need:

- Jupyter Notebook
- Python
- TensorFlow
- The original dataset
- The local development environment

The deployed application performs inference using the saved trained model.

---

# 🗃️ Model Files Required for Deployment

The Streamlit application primarily depends on:

```text
models/
│
├── plant_disease_best.keras
├── class_names.json
├── config.json
└── model_metadata.json
```

### `plant_disease_best.keras`

Contains the trained neural network used for inference.

### `class_names.json`

Maps numerical class IDs to human-readable disease names.

### `config.json`

Stores model and preprocessing configuration information.

### `model_metadata.json`

Stores model performance and related metadata used by the application.

---

# 🔐 Dataset and GitHub Considerations

The complete raw dataset is approximately:

```text
4.48 GB
```

and contains approximately:

```text
66,953 images
```

The raw image dataset is therefore not required for the deployed inference application.

The project uses:

```text
Dataset/master_images/
```

for model development and training, while the Streamlit application relies on the saved model artifacts.

The raw image directory should generally be excluded from the GitHub repository using `.gitignore`.

---

# ⚠️ Important Deployment Note

The trained model file must be available to the Streamlit application.

Before pushing the project to GitHub, check the size of:

```text
models/plant_disease_best.keras
```

If the model exceeds GitHub's normal individual-file limit, Git LFS or external model storage can be used.

---

# 🧠 Why EfficientNetB0?

EfficientNetB0 provides a practical balance between:

- Model capacity
- Computational requirements
- Inference speed
- Transfer-learning capability

Using an ImageNet-pretrained model allows the project to reuse general visual features and adapt them to plant disease classification through transfer learning and fine-tuning.

---

# 🔬 Why Transfer Learning?

Training a deep convolutional neural network completely from scratch requires:

- Large amounts of data
- Significant computational resources
- Longer training times
- Extensive hyperparameter tuning

Transfer learning provides pretrained visual representations that can be adapted to the plant disease domain.

The project therefore uses:

```text
ImageNet
    ↓
EfficientNetB0
    ↓
Plant Disease Classification
```

---

# 🔬 Why Fine-Tuning?

The features learned from ImageNet are general-purpose visual features.

Plant disease classification requires more specialized features such as:

- Leaf texture
- Lesion boundaries
- Disease spots
- Color changes
- Mosaic patterns
- Plant-specific symptoms

Fine-tuning allows later EfficientNetB0 layers to adapt these pretrained representations to the plant disease domain.

---

# 🎯 Why 281 Classes?

The dataset contains 281 distinct disease categories.

Therefore, the classification layer uses:

```text
Dense(281, activation="softmax")
```

Each output neuron corresponds to one disease class.

The model produces a probability distribution:

```text
Class 0   → Probability
Class 1   → Probability
Class 2   → Probability
...
Class 280 → Probability
```

The class with the highest probability becomes the Top-1 prediction.

---

# 🧠 Why Top-5 Prediction?

Plant diseases can have visually similar symptoms.

A model may therefore assign significant probability to several related classes.

Displaying Top-5 predictions provides more information than displaying only the highest-probability class.

The application therefore provides:

```text
Top-1 Prediction
       +
Top-5 Alternatives
       +
Probability Distribution
```

---

# 🔎 Why Grad-CAM?

Deep neural networks can make predictions while behaving like a black box.

Grad-CAM provides an additional visual interpretation.

It helps answer:

> Which regions of the image contributed to the prediction?

The project therefore combines:

```text
Prediction
+
Confidence
+
Top-5 Alternatives
+
Visual Explanation
```

rather than presenting only a disease name.

---

# 🧪 Model Limitations

Although the model achieves strong overall test-set performance, several limitations remain.

## 1. Fine-Grained Classification

The model distinguishes between 281 disease classes.

Some classes have visually similar symptoms, making them difficult to separate.

---

## 2. Dataset Dependency

Model performance depends on the characteristics of the training dataset.

Images that differ significantly from the training distribution may produce less reliable predictions.

---

## 3. Image Quality

Prediction quality can be affected by:

- Blur
- Poor lighting
- Severe background clutter
- Occlusion
- Very small leaves
- Multiple leaves
- Unusual camera angles

---

## 4. Confidence Is Not Certainty

The displayed probability is the model's predicted probability distribution.

A high confidence value does not guarantee that the diagnosis is biologically correct.

---

## 5. Field Conditions

The model was developed using image data and should not automatically be assumed to perform identically under all real-world agricultural conditions.

---

# ⚠️ Advisory & Disclaimer

This project is a **Deep Learning portfolio and research demonstration**.

The predictions generated by the application are probabilistic outputs from a neural network.

The system should **not be used as the sole basis for agricultural treatment, pesticide selection, crop-management decisions, or professional plant-disease diagnosis**.

For real-world agricultural decisions, predictions should be verified by a qualified agricultural professional or plant pathology expert.

---

# 🔮 Future Improvements

Potential future improvements include:

- Larger and more diverse field datasets
- Improved class balancing
- Hyperparameter optimization
- Advanced data augmentation
- Test-time augmentation
- Model calibration
- Confidence thresholding
- EfficientNetV2 experimentation
- MobileNet experimentation for lightweight deployment
- Vision Transformer experimentation
- Ensemble learning
- Advanced explainability methods
- SHAP-based analysis
- Improved Grad-CAM visualization
- Model monitoring
- Batch image prediction
- CSV-based prediction reports
- Prediction history
- Mobile-friendly UI improvements
- Automated model retraining
- CI/CD deployment
- Model versioning

---

# 🎓 Key Learning Outcomes

Through this project, I gained practical experience in:

## Data Handling

- Large-scale image datasets
- Dataset metadata
- Dataset splitting
- Class mapping
- Image dimension analysis

## Exploratory Data Analysis

- Distribution analysis
- Class analysis
- Dataset visualization
- Image visualization
- Dataset quality inspection

## Deep Learning

- TensorFlow
- Keras
- CNN architectures
- EfficientNetB0
- Transfer learning
- Fine-tuning
- Softmax classification

## Data Preprocessing

- Image decoding
- Image resizing
- TensorFlow `tf.data`
- Batching
- Prefetching
- Data augmentation

## Model Training

- Adam optimizer
- Sparse categorical crossentropy
- Early stopping
- Learning-rate scheduling
- Model checkpointing

## Model Evaluation

- Accuracy
- Top-5 accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Confidence analysis

## Explainable AI

- Grad-CAM
- TensorFlow GradientTape
- Feature-map visualization
- Model interpretation

## Deployment

- Streamlit
- Model serialization
- JSON metadata
- Inference pipelines
- Git
- GitHub
- Streamlit Community Cloud

---

# 💡 What This Project Demonstrates

This project demonstrates an end-to-end Deep Learning workflow rather than only model training.

```text
Dataset
   ↓
EDA
   ↓
Preprocessing
   ↓
Data Pipeline
   ↓
Model Architecture
   ↓
Transfer Learning
   ↓
Fine-Tuning
   ↓
Evaluation
   ↓
Explainable AI
   ↓
Model Serialization
   ↓
Web Application
   ↓
Deployment
```

The project combines:

```text
Data Science
+
Deep Learning
+
Computer Vision
+
Explainable AI
+
Application Development
+
Deployment
```

---

# 📸 Application Screenshots

Screenshots can be added after the final Streamlit deployment.

## 🏠 Streamlit Dashboard

The main application interface provides:

- Project overview
- Model information
- Image upload
- Prediction interface

Add screenshot:

```text
screenshots/dashboard.png
```

---

## 🌱 Disease Prediction

The application displays:

- Uploaded image
- Predicted disease
- Confidence percentage

Add screenshot:

```text
screenshots/prediction.png
```

---

## 📊 Top-5 Predictions

The application displays the five highest-probability disease classes.

Add screenshot:

```text
screenshots/top5-predictions.png
```

---

## 🧠 Grad-CAM Visualization

The application displays the original image alongside the Grad-CAM activation visualization.

Add screenshot:

```text
screenshots/gradcam.png
```

---

## 📈 Model Performance

The application displays:

```text
Test Accuracy       81.49%
Top-5 Accuracy      94.20%
Macro Precision     44.65%
Macro Recall        42.74%
Macro F1            42.28%
```

Add screenshot:

```text
screenshots/model-performance.png
```

---

# 📂 Repository Architecture

The project separates the major stages of the machine learning lifecycle:

```text
Dataset
   │
   ├── Raw Images
   ├── Metadata
   └── Dataset Splits
          │
          ▼
      Jupyter Notebook
          │
          ├── EDA
          ├── Preprocessing
          ├── Training
          ├── Fine-Tuning
          ├── Evaluation
          └── Grad-CAM
                  │
                  ▼
             Saved Models
                  │
                  ├── .keras
                  ├── class_names.json
                  ├── config.json
                  └── model_metadata.json
                           │
                           ▼
                    Streamlit Application
                           │
                           ▼
                 Streamlit Community Cloud
```

---

# 👨‍💻 Author

## Paulose Shaji

**B.Tech – Electronics & Communication Engineering**

Aspiring:

- Data Scientist
- Data Analyst
- AI/ML Engineer

### Focus Areas

- Python
- Data Science
- Machine Learning
- Deep Learning
- Computer Vision
- Explainable AI
- Power BI
- Streamlit
- AI Application Deployment

📍 Ernakulam, Kerala, India

---

# 🔗 Connect With Me

🌐 **Portfolio:**  
https://paulose-shaji.github.io

💼 **LinkedIn:**  
https://www.linkedin.com/in/paulose-shaji/

🐙 **GitHub:**  
https://github.com/paulose-shaji

---

# ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ **Star** on GitHub.

---

# 🌿 Project Summary

**Plant Disease Intelligence** is an end-to-end Deep Learning project that combines:

```text
66,953+ Images
        ↓
281 Disease Classes
        ↓
Computer Vision
        ↓
EfficientNetB0
        ↓
Transfer Learning
        ↓
Fine-Tuning
        ↓
81.49% Test Accuracy
        ↓
94.20% Top-5 Accuracy
        ↓
Grad-CAM Explainability
        ↓
Streamlit Application
        ↓
Cloud Deployment
```

Built with:

**Python • TensorFlow • Keras • EfficientNetB0 • Grad-CAM • Streamlit • GitHub 🚀**
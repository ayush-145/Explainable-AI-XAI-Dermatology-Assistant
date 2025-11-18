# DermaNet-X: Explainable AI (XAI) Dermatology Assistant

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.0-61DAFB?logo=react&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

**DermaNet-X** is a full-stack, production-ready medical diagnostic system designed to classify skin lesions with high sensitivity while providing **visual explainability** to clinicians. 

Unlike standard "black box" classifiers, this system uses **Grad-CAM** to highlight lesion focal points and calculates **prediction entropy** to flag uncertain cases for human review.

---

## 🧠 Key Features

* **Advanced Model Architecture**: Fine-tuned **EfficientNet-B4** (trained on HAM10000), optimized for medical imaging features.
* **Explainable AI (XAI)**: Integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate heatmaps, allowing doctors to verify *why* the model made a prediction.
* **Uncertainty Quantification**: Calculates entropy-based uncertainty scores. If the model is "confused" (high entropy), it flags the diagnosis as **High Risk** regardless of the class probability.
* **Robust Preprocessing**: Implements a **Digital Hair Removal** algorithm using OpenCV morphological operations to clean lesion images before inference.
* **Class Imbalance Handling**: Trained using **Weighted Focal Loss** to prioritize rare but dangerous classes (e.g., Melanoma) over common benign moles.
* **Full-Stack Deployment**:
    * **Backend**: High-performance **FastAPI** microservice.
    * **Frontend**: Interactive **React.js** dashboard for clinicians.
    * **DevOps**: Fully containerized with **Docker** and **Docker Compose** for one-command deployment.

---

## 🏗️ System Architecture

```mermaid
graph LR
    User[Clinician] -->|Upload Image| UI[React Frontend]
    UI -->|POST /predict| API[FastAPI Backend]
    
    subgraph "AI Inference Engine"
        API -->|Raw Bytes| Pre[OpenCV Hair Removal]
        Pre -->|Clean Tensor| Model[EfficientNet-B4]
        Model -->|Logits| GradCAM[Grad-CAM Generator]
        Model -->|Softmax| Entropy[Uncertainty Calculator]
    end
    
    GradCAM -->|Heatmap Image| API
    Entropy -->|Risk Flag| API
    API -->|JSON Response| UI

    

# AKA Marine Oil Spill Detection System

An end-to-end deep learning project for marine oil spill detection, localization (segmentation), and web deployment. This repository contains the source code for the U-Net segmentation model, ResNet classification model, and a Flask application for interactive inference.

Created by: Kaarthik Nalla

---

## 🚀 Project Overview

This system addresses the challenge of monitoring maritime environments by performing two key machine learning tasks:

1.  **Classification:** Determines the probability of oil presence in an entire image (**ResNet50**).
2.  **Segmentation:** Accurately maps the precise pixel boundaries of the oil slick (**U-Net**).

The system includes academic evaluation metrics calculated on a dedicated Test Set and is deployed via a local Flask web application.

---

## 📊 Final Test Set Performance

| Metric | Segmentation (U-Net) | Classification (ResNet) |
| :--- | :---: | :---: |
| **IoU (Jaccard Index)** | **0.7103** | N/A |
| **Dice Coefficient** | **0.8549** | N/A |
| **AUC (ROC)** | N/A | **1.0000** |
| **F1 Score** | 0.8549 | 1.0000 |

---

## 🛠️ Project Structure

<img width="1076" height="621" alt="image" src="https://github.com/user-attachments/assets/2781dd42-cfd3-4cd6-a4a5-184a2e1ba51b" />

---

## ⚙️ Installation & Usage

### Prerequisites

1.  Python 3.8+
2.  Git and GitHub Desktop/CLI
3.  All packages listed in `requirements.txt`

### Steps to Run Locally

1.  **Clone the Repository** (If applicable):
    `git clone [Your Repository URL]`
    `cd marine-oil-spill`

2.  **Setup Environment**
    `python -m venv venv`
    `.\venv\Scripts\activate`  (Use `source venv/bin/activate` on Linux/macOS)
    `pip install -r requirements.txt`

3.  **Prepare Data**
    `python src/mask_generation.py`
    `python src/split_data.py`

4.  **Train Models**
    `python -m src.train_seg`
    `python -m src.train_cls`

5.  **Run Final Evaluation**
    `python -m src.evaluate`
    (Saves metrics to `test_metrics_results.json` and plot to `journal_metrics.png`)

---

## 🌐 Web Deployment

1.  **Launch the App:**
    `python flask_app/app.py`

2.  **Access:** Open your web browser and navigate to the displayed link (usually `http://127.0.0.1:5000/`) to view the interactive detector.

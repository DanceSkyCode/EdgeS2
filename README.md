# EdgeS²: Lightweight Edge-Disentangled Spiral Scanning Learning for  
## 3D CT Difficult Laryngoscopy Prediction

This repository provides the official PyTorch implementation of **EdgeS²**, a lightweight edge-disentangled spiral scanning framework for predicting **difficult laryngoscopy (DL)** from **3D head-and-neck CT volumes**.

EdgeS² is designed to achieve high predictive performance with extremely low
computational cost, making it suitable for deployment in resource-constrained
clinical environments.

---

## Overview

Difficult laryngoscopy (DL) is a major challenge in anesthetic airway management
and a significant source of perioperative risk. While 3D head-and-neck CT enables
objective preoperative DL assessment, existing learning-based methods are often
limited by insufficient global anatomical modeling, implicit boundary
representation, and high model complexity.

We propose **EdgeS²**, a lightweight edge-disentangled spiral scanning framework
for DL prediction from 3D CT volumes. EdgeS² adopts a local-to-global design, where
a learnable decomposition module reduces spatial redundancy while preserving
structural information. Critical airway boundaries are encoded via shallow
edge-aware features, and long-range anatomical dependencies across axial,
sagittal, and coronal planes are captured using a 3D spiral scanning Mamba
mechanism. A global triple cross-attention module integrates edge-aware,
non-edge-aware, and semantic representations.

Experiments on an in-house dataset of 499 patients demonstrate that EdgeS²
achieves state-of-the-art performance (AUC 0.908, F1-score 0.833) with only
0.422M trainable parameters.

---

## Method Highlights

- **Edge-Disentangled Representation Learning**  
  Explicitly separates boundary-sensitive and non-boundary features.

- **Learnable Decomposition Module**  
  Reduces spatial redundancy while preserving anatomical structure.

- **3D Spiral Scanning Mamba (S²Mamba)**  
  Implicitly captures long-range 3D anatomical dependencies with linear
  computational complexity, avoiding explicit multi-view feature extraction
  and fusion.

- **Triple Cross-Attention Integration**  
  Jointly aggregates edge-aware, non-edge-aware, and deep semantic features.

---

## Repository Structure
```
├── model.py # EdgeS² network architecture
├── dataset.py # Dataset loading and preprocessing
├── train.py # Training and validation script
├── requirements.txt # Required Python dependencies
│
├── framework.png # Network architecture illustration
├── visualization.png # Saliency and interpretability visualization
├── analysis.png # Quantitative analysis results
│
├── 4666073.nii.gz # Example 3D CT volume
├── 4671339.nii.gz # Example 3D CT volume
```
---

## Visualization and Analysis

- **framework.png**  
  Overall architecture of the proposed EdgeS² framework.

- **visualization.png**  
  2D and 3D saliency maps highlighting anatomically meaningful airway boundaries.

- **analysis.png**  
  Performance analysis and experimental results.

- **\*.nii.gz**  
  Example CT volumes provided for demonstration and testing purposes.

---

## Requirements

Please install the required dependencies using:
pip install -r requirements.txt


It is strongly recommended to **strictly follow the provided dependency
versions**, as deviations may lead to inconsistent results.

---

## Training and Evaluation

To train and evaluate the EdgeS² model, run:
```
python train.py
```
# MRI-to-sCT Generation using CycleGAN (Head & Neck)

This project focuses on generating synthetic CT (sCT) images from T1-weighted Head & Neck MRI using a CycleGAN-based framework.

The ultimate goal is to enable MRI-only radiotherapy planning by generating high-quality sCT volumes and evaluating them quantitatively and anatomically.

---

## Role of This Repository (Current Progress)

The 3D multi-organ segmentation models implemented here are **not the final objective**, but rather a supporting evaluation component for the main CycleGAN pipeline.

We implemented and compared three 3D segmentation architectures:

- **UNet 3D**
- **Attention UNet 3D**
- **UNet++ 3D (Lightweight Nested Variant)**

These models are trained to segment six critical Head & Neck organs:

1. Bone Mandible  
2. Brainstem  
3. Spinal Cord  
4. Parotid Left  
5. Parotid Right  
6. Oral Cavity  

The purpose of these segmentation models is to:

- Evaluate anatomical consistency of generated sCT images  
- Compare organ-wise Dice scores between real CT and generated sCT  
- Provide structural validation metrics for the CycleGAN output  

---

## Project Status

The segmentation work in this repository represents approximately **15% of the overall project**.

The main project components include:

- MRI → sCT generation using CycleGAN  
- Image intensity validation (HU analysis)  
- Anatomical validation using segmentation models  
- Clinical feasibility evaluation  

Currently, this repository contains:

- Patch-based 3D training pipeline  
- Sliding window full-volume inference  
- Quantitative Dice evaluation  
- Comparative study of UNet variants  

The segmentation module serves as a validation backbone for the full MRI-to-sCT synthesis pipeline.


---

# Project Setup

## 1. Create Virtual Environment

```bash
python -m venv venv
```

## 2. Activate Environment
Windows
```bash
venv\Scripts\activate
```
Linux / Mac
```bash
source venv/bin/activate
```

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```



## Training

- Training is patch-based using:
- Patch Size: 80 × 80 × 80
- Balanced foreground sampling
- Mixed Precision Training (AMP)
- Dice Loss + Cross Entropy Loss

## Training the Models

Run the following notebooks/scripts in order:

🔹 UNet 3D
```bash
notebooks/unet_training.ipynb
```
🔹 Attention UNet 3D
```bash
notebooks/attention_unet_training.ipynb
```
🔹 UNet++ 3D (Lightweight)
```bash
notebooks/unetpp_training.py
```

## Model Checkpoints
All trained models and checkpoints are automatically saved inside:
```bash
experiments/
```
Each model has its own subfolder (e.g., unet_fold0, attention_unet_fold0, etc.).

## Evaluation Notebook

Full-volume evaluation and visualization notebooks are available inside:
```bash
evaluation/
```
Use the appropriate evaluation notebook to:

Load trained checkpoints
Perform sliding-window inference
Compute full-volume Dice scores
Visualize predictions and difference maps
  
## Evaluation for now
1. Patch Evaluation
2. Full-volume evaluation is performed using sliding-window inference:
Patch Size: 80
Stride: 60

# Progress

Full-volume evaluation
| Model          | Mandible | Brainstem | Spinal Cord | Parotid L | Parotid R | Oral Cavity |
| -------------- | -------- | --------- | ----------- | --------- | --------- | ----------- |
| UNet           | 0.75     | 0.68      | ~0.00       | 0.58      | 0.74      | 0.76        |
| Attention UNet | **0.84** | **0.77**  | ~0.00       | 0.67      | 0.74      | **0.79**    |
| UNet++         | 0.76     | 0.72      | **0.56**    | **0.70**  | 0.72      | 0.38        |





  


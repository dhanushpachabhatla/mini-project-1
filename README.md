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
Python 3.10.11 was working fine without errors (if getting errors switch to this)
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

### Full-volume evaluation


| #     | Model                             | Architecture                   | Base Filters | Patch Size | Patches / Case | FG Sampling | Augmentation           | Batch Size | Loss                   | Optimizer             | Inference                                                 |
| ----- | --------------------------------- | ------------------------------ | ------------ | ---------- | -------------- | ----------- | ---------------------- | ---------- | ---------------------- | --------------------- | --------------------------------------------------------- |
| **1** | **UNet 3D (Baseline)**            | Standard 3D U-Net              | 16           | 80³        | 6              | 0.5         |  Disabled             | 2          | Cross-Entropy          | Adam                  | Sliding window (no Gaussian)                              |
| **2** | **Attention UNet 3D**             | 3D Attention U-Net             | 16           | 80³        | 6              | 0.5         |  Disabled             | 2          | Cross-Entropy          | Adam                  | Sliding window (no Gaussian)                              |
| **3** | **UNet++ 3D**                     | Lightweight Nested U-Net       | 16           | 80³        | 6              | 0.5         |  Disabled             | 2          | Cross-Entropy          | Adam                  | Sliding window (no Gaussian)                              |
| **4** | **nnUNet-style (Initial)**        | Residual 3D U-Net              | 24           | 80³        | 6              | 0.5         |  Disabled             | 2          | Cross-Entropy          | Adam                  | Sliding window (simple averaging)                         |
| **5** | **Final nnUNet Pipeline** | Residual nnUNet-style 3D U-Net | 24           | **96³**    | **12**         | **0.6**     |  TorchIO Augmentation | 2          | **Dice + CE (Hybrid)** | **AdamW + Cosine LR** | **Sliding window + Gaussian weighting + Post-processing** |


| Model          | Mandible | Brainstem | Spinal Cord | Parotid L | Parotid R | Oral Cavity |
| -------------- | -------- | --------- | ----------- | --------- | --------- | ----------- |
| UNet           | 0.75     | 0.68      | ~0.00       | 0.58      | 0.74      | 0.76        |
| Attention UNet | **0.84** | **0.77**  | ~0.00       | 0.67      | 0.74      | **0.79**    |
| UNet++         | 0.76     | 0.72      | **0.56**    | **0.70**  | 0.72      | 0.38        |





### Final nnUNET 
| Organ ID | Organ         | Mean Dice  | Std Dev |
| -------- | ------------- | ---------- | ------- |
| 1        | Bone Mandible | **0.8928** | 0.0275  |
| 2        | Brainstem     | **0.8015** | 0.0210  |
| 3        | Spinal Cord   | **0.7350** | 0.0272  |
| 4        | Parotid Left  | **0.4705** | 0.0663  |
| 5        | Parotid Right | **0.2422** | 0.1312  |
| 6        | Oral Cavity   | **0.8251** | 0.0580  |


### Custom Loss Model Evaluation Comparison

### lambda = 0.4

| Model & Configuration | Class 1 | Class 2 | Class 3 | Class 4 (Parotid L) | Class 5 (Parotid R) | Class 6 | Mean Dice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (Initial Weights)**<br>`[0.05, 1, 1, 1.2, 1.8, 1.8, 1]` | 0.8960 | 0.8017 | 0.7107 | 0.8102 | 0.7981 | 0.8434 | 0.8100 |
| **Model 2 (Revised Weights)**<br>`[0.05, 1, 1.2, 1.5, 2, 2, 1]` | **0.9132** | **0.8017** | **0.7583** | 0.8170 | 0.7837 | 0.8353 | **0.8182** |
| **Model 3 (Spinal Favored Patching)**<br>`[0.05, 1, 1.3, 1.8, 2, 2, 1]` | 0.9119 | 0.7586 | 0.7461 | **0.8180** | **0.8033** | **0.8556** | 0.8156 |

**Observations**:
* **Model 2 (Revised Weights)** achieved the highest overall **Mean Dice (0.8182)** and performed best on Classes 1, 2, and 3.
* **Model 3 (Spinal Favored Patching)** sacrificed some performance on Class 2, but achieved the best results for the Parotid glands (Classes 4 & 5) and Class 6.


### Model Evaluation Comparison (Lambda = 0.6)

| Model & Configuration | Class 1 | Class 2 | Class 3 | Class 4 (Parotid L) | Class 5 (Parotid R) | Class 6 | Mean Dice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1**<br>`[0.05, 1, 1, 1.2, 1.8, 1.8, 1]` | 0.9047 ± 0.0235 | **0.8100 ± 0.0349** | 0.7394 ± 0.0585 | 0.7768 ± 0.0279 | 0.7894 ± 0.0770 | 0.8007 ± 0.0609 | 0.8035 |
| **Model 2**<br>`[0.05, 1, 1.2, 1.5, 2, 2, 1]` | **0.9142 ± 0.0266** | 0.8083 ± 0.0324 | **0.7553 ± 0.0526** | **0.8094 ± 0.0298** | **0.7975 ± 0.0571** | **0.8559 ± 0.0362** | **0.8234** |

**Observations (λ=0.6)**:
* **Model 2** significantly outperforms Model 1 across almost all classes, achieving a higher overall Mean Dice (0.8234).
* The Parotid glands (Classes 4 & 5) and Class 6 show noticeable improvements in Model 2 upon revising the class weights.


  


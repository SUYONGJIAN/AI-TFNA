# AI-TFNA: AI-Thyroid Fine Needle Aspiration Cytological Classification

> Multimodal deep learning for thyroid nodule cytological diagnosis aligned with The Bethesda System (TBS).

---

## 🔗 Quick Links
- Paper: [Cytological classification Diagnosis for thyroid nodules via multimodal model deep learning](<http://doi.org/10.1002/advs.202511369>)
- GitHub: https://github.com/SUYONGJIAN/AI-TFNA

---

## 📌 Overview
AI-TFNA is a multimodal deep-learning pipeline that automatically classifies thyroid fine-needle aspiration (FNA) cytology slides according to The Bethesda System (TBS).  
The framework integrates nuclear morphology, cellular phenotype and slide-level context to deliver highly accurate cytological diagnosis and BRAF-mutation prediction.

---

## 🏗️ Architecture
| Module | Role |
|--------|------|
| **SEG-DETECT** | Nuclear segmentation & morphological feature extraction (XFPN-U-Net backbone) |
| **VAN-tiny**   | Single-cell / cluster-cell phenotype classification |
| **XGBoost**    | Slide-level TBS category prediction |

---

## ✨ Key Features
* **TBS-compliant**: six-tier classification aligned with The Bethesda System  
* **State-of-the-art segmentation**: XFPN-U-Net with **98.39 %** single-cell recall and **96.70 %** cluster-cell recall  
* **Color-robust**: built-in Image Appearance Migration (IAM) module handles inter-lab staining variance  
* **Mutation insight**: optional BRAF-V600E mutation probability output  
* **End-to-end**: from raw WSIs to TBS report in one pipeline

---

## 🚀 Getting Started
```bash
git clone https://github.com/SUYONGJIAN/AI-TFNA.git
cd AI-TFNA

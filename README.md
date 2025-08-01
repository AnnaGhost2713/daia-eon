![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Status](https://img.shields.io/badge/status-research-brightgreen)

# 🛡️ DAIA-EON: Privacy-Preserving Anonymization of German Energy-Retail Customer Emails: A PII Detection and Synthetic Data Augmentation Framework

This project explores automated anonymization of German customer communication data through Named Entity Recognition (NER). It combines synthetic data generation, fine-tuned language models, and comprehensive evaluation metrics to build a robust and privacy-compliant anonymization pipeline.

---

## 📘 Project Description

Sensitive customer data in emails (e.g., names, addresses, meter numbers) must be anonymized for analysis, training, and sharing. However, building performant anonymization models requires large amounts of annotated data—data that is often unavailable due to privacy constraints.

**DAIA-EON** tackles this challenge through:
- High-quality **synthetic data generation** using paraphrasing and realistic entity injection.
- Evaluation against a **manually annotated gold standard**.
- Training and comparison of **NER models** (Piiranha, spaCy, Gemini).
- Support for **granular entity categories** (e.g., contract number, payment, IBAN).

This project was developed in collaboration with **E.ON** as part of the university course **Data Analytics in Applications**.

---

## 🗂️ Table of Contents

- [Project Description](#project-description)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Team & Credits](#team--credits)
- [License](#license)

---

## 📂 Repository Structure

```bash
├── archive
│   ├── backup
│   ├── old_data
│   └── old_workbooks
├── data
│   ├── excel_manual_labeling
│   ├── golden_dataset_anonymized
│   ├── original
│   ├── synthetic
│   └── testing
├── notebooks
│   ├── 1_data_preparation
│   ├── 2_synthetic_data_generation
│   ├── 3_model_training_and_testing
│   └── data
├── README.md
├── requirements.txt
└── src
    └── __init__.py
```

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/AnnaGhost2713/daia-eon.git
cd daia-eon
```

### 2. (Recommended) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install project dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

This project is divided into three main stages:

### 1. Data Preparation

Located in `notebooks/1_data_preparation`, this stage:

- Prepares and aligns the golden dataset.
- Splits it into training and test sets.
- Outputs span-based JSON for NER training.

### 2. Synthetic Data Generation

Located in `notebooks/2_synthetic_data_generation`, this stage:

- Creates labeled synthetic emails using two approaches:
  - **option_a**: Paraphrased templates with Faker entity injection.
  - **option_b**: Balanced sampling based on entity label frequency.
- Evaluates data quality using:
  - Perplexity
  - BERTScore
  - kNN classification
  - Downstream NER F1

### 3. Model Training & Testing

Located in `notebooks/3_model_training_and_testing`, this stage:

- Fine-tunes and evaluates models (e.g., Piiranha, spaCy, Gemini).
- Compares model performance on golden and synthetic test sets.


**Tip:**  
For each stage, follow the step-by-step instructions in the corresponding Jupyter notebooks.  
You can run the notebooks directly with:

```bash
jupyter notebook <path-to-notebook>
```
---

## 👥 Team & Credits

This project was developed as part of the "Data Analytics in Applications" course in collaboration with **E.ON**.

**Project Contributors:**
- **Anna-Maria Geist** ([AnnaGhost2713](https://github.com/AnnaGhost2713)) 
- **Moritz Gärtner** ([moritzgaertner](https://github.com/moritzgaertner)) 
- **Timon Martens** ([timonmartens](https://https://github.com/timonmartens)) 
- **Nicholas Hecker** ([ThisIsWallE](https://github.com/ThisIsWallE)) 

Special thanks to our mentors at **E.ON** for their feedback and data support.

---

## 📄 License

License to be determined.  
If you plan to use or contribute to this project, please contact the authors for permissions or clarifications.

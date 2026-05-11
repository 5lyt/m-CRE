# m-CRE

m-CRE is a deep learning framework for high/low plant gene expression classification by integrating regulatory sequence information and mRNA half-life-related features. The framework combines TSS- and TTS-flanking sequence inputs with transcript-structure-derived features and supports ablation analysis, chromosome-wise validation, and DeepSHAP-based model interpretation.

## Overview

This repository contains the source code used for:

- extracting TSS- and TTS-flanking regulatory sequences;
- calculating mRNA half-life-related features;
- constructing high/low gene expression labels;
- training sequence-only, feature-only, and fused deep learning models;
- performing chromosome-wise validation;
- conducting DeepSHAP-based interpretability analysis;
- generating partial figures and result summaries for publication.

## Species and tissues

The model was evaluated on leaf and root datasets from four representative plant species:

- *Arabidopsis thaliana*;
- *Solanum lycopersicum*;
- *Sorghum bicolor*;
- *Zea mays*.

## Repository structure

```text
m-CRE/
├── code/
│   ├── data_deal/          # Data preprocessing scripts
│   ├── model/              # Model training and validation scripts
│   └── interpretation/     # DeepSHAP and visualization scripts
│
├── data/
│   ├── example/            # Representative processed example data
│   └── metadata/           # Metadata files, accession lists, and validation information
│
├── results/                # Selected result tables and figures
├── R/                      # WGCNA-related scripts and auxiliary analyses
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
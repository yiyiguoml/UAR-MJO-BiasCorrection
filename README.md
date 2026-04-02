# Unified Attention Recurrent Neural Network for Bias Correction of MJO Prediction

This repository contains the code and archived data associated with the paper:

**Unified Attention Recurrent Neural Network for Bias Correction of MJO Prediction**

## Overview

The Madden–Julian Oscillation (MJO) is a dominant mode of tropical intraseasonal variability and an important source of subseasonal forecast skill. This repository accompanies our JAMES manuscript and contains the implementation of a deep learning post-processing framework for bias correction of MJO forecasts based on a unified attention recurrent neural network (UAR).

The model is designed to correct systematic errors in predicted RMM indices and uses a phase-regularized training objective to improve both amplitude and phase accuracy.

## Repository Structure

```text
MJO_JAMES_CODE/
├── data_preprocessing/
│   └── data_preprocessing.py
├── main/
│   └── main.py
├── raw_data/
├── RMM_DATA_RETRIVED_FROM_S2S/
│   ├── BOM/
│   │   ├── rmm1.nc
│   │   └── rmm2.nc
│   ├── CNRM/
│   │   ├── rmm1.nc
│   │   └── rmm2.nc
│   ├── JMA/
│   │   ├── rmm1.nc
│   │   └── rmm2.nc
│   └── observed/
│       ├── rmm1.nc
│       └── rmm2.nc
├── preprocessing_data/
├── outputs/
└── README.md
```

## Data

The folder `RMM_DATA_RETRIVED_FROM_S2S/` contains the raw RMM index files used in this study for the three forecast systems and the observational reference data:

- `BOM/`
- `CNRM/`
- `JMA/`
- `observed/`

Each subfolder contains:

- `rmm1.nc`
- `rmm2.nc`

These files are used for preprocessing, model training, and evaluation.

## Original Data Sources

The Subseasonal to Seasonal (S2S) Prediction Project Database is used in this study. ERA-Interim reanalysis data are used as observational references for the MJO indices.

Please acknowledge the original data sources when using these materials:

- Vitart et al. (2017), for the S2S Prediction Project Database
- Dee et al. (2011), for ERA-Interim reanalysis

## Code

The repository includes:

- `data_preprocessing/data_preprocessing.py` for preparing datasets
- `main/main.py` for model training and evaluation

## Reproducibility

This repository is provided to support reproducibility of the experiments reported in the associated manuscript. It contains the archived code and data used in the study.

## Citation

If you use this repository, please cite the associated paper.

## Acknowledgments

We acknowledge the use of the Subseasonal to Seasonal (S2S) Prediction Project Database and ERA-Interim reanalysis data in this study.

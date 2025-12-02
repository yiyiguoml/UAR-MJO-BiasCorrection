# MJO Ensemble Bias Correction using Attention-based RNN

This repository contains the code for bias-correcting Madden-Julian Oscillation (MJO) ensemble forecasts using a multi-head attention RNN model.

## Overview

The Madden-Julian Oscillation (MJO) is the dominant mode of intraseasonal variability in the tropical atmosphere. Accurate MJO forecasting is critical for subseasonal-to-seasonal (S2S) prediction, but ensemble forecasts often suffer from systematic biases. This project implements a deep learning approach to correct these biases using historical forecast-observation pairs.

### Key Features

- **Multi-head Attention RNN**: Processes ensemble member sequences with attention mechanism
- **Bivariate Error Metrics**: Uses BMSE (Bivariate Mean Squared Error) for phase-amplitude prediction
- **Automatic Hyperparameter Selection**: Grid search over cosine loss weight with early stopping
- **Reproducible Results**: Multiple independent runs with fixed random seeds

## Repository Structure

```
MJO_JAMES_CODE/
├── data_preprocessing/
│   └── data_preprocessing.py    # Prepare train/validation/test datasets
├── main/
│   └── main.py                  # Training script with model definition
├── raw_data/
│   ├── s2s_data.nc             # S2S ensemble forecasts (input)
│   └── observed_data.nc        # ERA-Interim observations (target)
├── preprocessing_data/          # Processed .npy files (generated)
├── outputs/                     # Trained model checkpoints (generated)
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- conda or pip for package management

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda install numpy pandas xarray netCDF4 scikit-learn
```

## Usage

### 1. Data Preprocessing

First, prepare the training/validation/test datasets from raw netCDF files:

```bash
cd data_preprocessing
python data_preprocessing.py
```

This script:
- Loads S2S ensemble forecasts and ERA-Interim observations
- Creates samples for each forecast initialization and lead time
- Computes derived features (amplitude, phase, seasonal encoding)
- Splits data into training (390 samples), validation (150 samples), and test sets
- Outputs processed data to `preprocessing_data/` directory

**Expected output**:
- `target_y{year}.train.npy` (1996-2010)
- `target_y{year}.validate.npy` (1996-2010)
- `target_y{year}.test.npy` (1981-2010)

### 2. Model Training

Train the bias correction model:

```bash
cd main
python main.py
```

The training script:
- Trains models for each test year (1996-2010)
- Performs hyperparameter search over cosine loss weight λ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
- Uses early stopping based on validation BMSE
- Runs 5 independent trials for statistical robustness
- Saves best models to `outputs/Ours_ic_Nosquare/{run}/{year}.pth`

**Training time**: ~2-4 hours per run on A100 GPU

### 3. Model Evaluation

The training script automatically evaluates on test data and prints:
- BMSE amplitude error (BMSEa)
- BMSE phase error (BMSEb)
- Total BMSE = BMSEa + BMSEb
- Comparison with uncorrected S2S forecasts

## Model Architecture

### Input Features

Each sample contains 34 time steps (initial condition + 33 forecast leads) with 4 features:
1. RMM1 (Real component of MJO index)
2. RMM2 (Imaginary component of MJO index)
3. MJO amplitude
4. Forecast lead time (0-33 days)

### Network Architecture

```
Input (batch, 34, 4)
    ↓
RNN (2 layers, hidden_size=40)
    ↓
Multi-head Attention (2 heads)
    ↓
Attention Pooling
    ↓
Fully Connected Layer
    ↓
Output (batch, 2)  # Predicted RMM1, RMM2
```

### Loss Function

Combined MSE and cosine similarity loss:

```
L = MSE(RMM1, RMM2) + λ × (1 - cosine_similarity)
```

where λ is selected via validation set performance.

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_size` | 40 | RNN hidden state dimension |
| `num_layers` | 2 | Number of RNN layers |
| `num_heads` | 2 | Multi-head attention heads |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `total_epoch` | 2000 | Maximum training epochs |
| `patience` | 20 | Early stopping patience |

## Data

### Input Data Format

**S2S Forecasts** (`s2s_data.nc`):
- Variables: RMM1, RMM2
- Dimensions: time, lead (0-33 days)
- Period: 1981-2010
- Source: Bureau of Meteorology (BOM) ensemble

**Observations** (`observed_data.nc`):
- Variables: RMM1, RMM2
- Dimensions: time
- Period: 1981-2010
- Source: ERA-Interim reanalysis

### Train/Validation/Test Split

- **Training**: 15 years of historical data before target year (390 samples)
- **Validation**: Uniformly sampled from training period (150 samples)
- **Test**: Target year forecasts (varies by year)
- **Test years**: 1996-2010 (15 years)

## Bivariate Mean Squared Error (BMSE)

BMSE decomposes forecast error into amplitude and phase components:

**Amplitude error (BMSEa)**:
```
BMSEa = mean((A_forecast - A_observed)²)
```

**Phase error (BMSEb)**:
```
BMSEb = mean(2 × A_forecast × A_observed × (1 - cos(θ_forecast - θ_observed)))
```

where `A = sqrt(RMM1² + RMM2²)` and `θ = atan2(RMM2, RMM1)`

## Results

Expected performance (averaged over 5 runs, 1996-2010):

| Model | BMSEa | BMSEb | Total BMSE |
|-------|-------|-------|------------|
| S2S Uncorrected | ~X.XX | ~X.XX | ~X.XX |
| Bias Corrected | ~X.XX | ~X.XX | ~X.XX |

*(Fill in with your actual results)*

## Citation

If you use this code for research, please cite:

```bibtex
@article{YourName2024,
  title={MJO Ensemble Bias Correction using Attention-based RNN},
  author={Your Name et al.},
  journal={Journal of Advances in Modeling Earth Systems (JAMES)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- S2S forecast data provided by Bureau of Meteorology (BOM)
- ERA-Interim reanalysis data from ECMWF
- Computational resources provided by [Your Institution]

## Contact

For questions or issues, please contact:
- Email: [your.email@institution.edu]
- GitHub Issues: [repository-url]/issues

## References

1. Wheeler, M. C., & Hendon, H. H. (2004). An all-season real-time multivariate MJO index. *Monthly Weather Review*, 132(8), 1917-1932.
2. Vitart, F., & Robertson, A. W. (2018). The sub-seasonal to seasonal prediction project (S2S) and the prediction of extreme events. *npj Climate and Atmospheric Science*, 1(1), 3.

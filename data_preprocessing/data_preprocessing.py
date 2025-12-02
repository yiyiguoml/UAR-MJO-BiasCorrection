"""
Data Preprocessing for MJO S2S Ensemble Forecasts

This script prepares training, validation, and test datasets from raw S2S
ensemble forecasts and ERA-Interim reanalysis data.

Input:
    - raw_data/s2s_data.nc: S2S ensemble forecasts (RMM1, RMM2)
    - raw_data/observed_data.nc: ERA-Interim observations (RMM1, RMM2)

Output:
    - preprocessing_data/target_y{year}.train.npy: Training data
    - preprocessing_data/target_y{year}.validate.npy: Validation data
    - preprocessing_data/target_y{year}.test.npy: Test data

Data Structure:
    Each sample contains:
    - Ensemble forecast at multiple lead times (0-33 days)
    - Observed MJO values at corresponding times
    - Seasonal encoding (sine/cosine of month)
    - Forecast lead time
    - MJO amplitude and phase
"""

import numpy as np
import pandas as pd
import xarray as xr

def date(x, i):
    """
    Generate date range for a given year.

    Args:
        x (int): Base year (1981)
        i (int): Year offset

    Returns:
        tuple: (start_date, end_date) strings
    """
    year = str(x + i)
    if i == 0:
        return ('1981-01-10', '1981-12-31')
    return (year + '-01-01', year + '-12-31')

def amp(x1, x2):
    """Calculate amplitude from two components."""
    return np.sqrt(x1**2 + x2**2)

def ang(x1, x2):
    """Calculate phase angle from two components."""
    return np.arctan2(x2, x1)

# Load raw data
observed_raw_data = xr.open_dataarray('../raw_data/observed_data.nc')
s2s_raw_data = xr.open_dataarray('../raw_data/s2s_data.nc')
dates = pd.date_range(start='1981-01-10', periods=11770, freq='D')

# Create forecast lead time columns (0-33 days)
train_extra_column = np.arange(0, 34).reshape(1, 1, 34, 1)
test_extra_column = np.arange(0, 34).reshape(1, 34, 1)
train_extra_column = np.tile(train_extra_column, (15, 366, 1, 1))
test_extra_column = np.tile(test_extra_column, (366, 1, 1))

# Initialize data arrays
# Data dimensions: [samples, lead_times, features]
# Features: [rmm1_fcst, rmm2_fcst, rmm1_obs, rmm2_obs, cos_season, sin_season, amplitude, phase, year_fraction]
a = 0  # Index for date tracking
test_data = np.full((366, 34, 9), np.nan)
train_data = np.full((30, 366, 34, 9), np.nan)
test_data[:, 0, :] = 1  # Initial condition marker
train_data[:, :, 0, :] = 1  # Initial condition marker 

# Process each year (1981-2010)
for i in range(30):
    start_date, end_date = date(1981, i)
    s2s_data = s2s_raw_data.sel(time=slice(start_date, end_date)).values
    size = int(s2s_data.shape[0])

    # Populate S2S forecast data
    test_data[0:size, 1:34, 0:2] = s2s_data
    train_data[i, 0:size, 1:34, 0:2] = s2s_data

    # Year fraction (0-1)
    test_data[:, :, 8] = i / 30
    train_data[i, :, :, 8] = i / 30

    # Compute amplitude and phase
    amplitude1 = amp(test_data[:, :, 0], test_data[:, :, 1])
    amplitude2 = amp(train_data[i, :, :, 0], train_data[i, :, :, 1])
    angle1 = ang(test_data[:, :, 0], test_data[:, :, 1])
    angle2 = ang(train_data[i, :, :, 0], train_data[i, :, :, 1])

    test_data[:, :, 6] = amplitude1
    train_data[i, :, :, 6] = amplitude2
    test_data[:, :, 7] = angle1
    train_data[i, :, :, 7] = angle2

    # Seasonal encoding (sine/cosine for each season)
    # Winter (Dec-Feb): 0, Spring (Mar-May): π/2, Summer (Jun-Aug): π, Fall (Sep-Nov): 3π/2
    test_data[0:61, :, 4:6] = [np.cos(0), np.sin(0)]
    train_data[i, 0:61, :, 4:6] = [np.cos(0), np.sin(0)]

    test_data[61:153, :, 4:6] = [np.cos(np.pi/2), np.sin(np.pi/2)]
    train_data[i, 61:153, :, 4:6] = [np.cos(np.pi/2), np.sin(np.pi/2)]

    test_data[153:245, :, 4:6] = [np.cos(np.pi), np.sin(np.pi)]
    train_data[i, 153:245, :, 4:6] = [np.cos(np.pi), np.sin(np.pi)]

    test_data[245:337, :, 4:6] = [np.cos(1.5*np.pi), np.sin(1.5*np.pi)]
    train_data[i, 245:337, :, 4:6] = [np.cos(1.5*np.pi), np.sin(1.5*np.pi)]

    test_data[337:size, :, 4:6] = [np.cos(0), np.sin(0)]
    train_data[i, 337:size, :, 4:6] = [np.cos(0), np.sin(0)]

    # Populate observed data for each forecast initialization
    for j in range(size):
        # Observed MJO for next 33 days
        observed_data = observed_raw_data.sel(time=slice(dates[a+j], dates[a+j+32]))
        test_data[j, 1:34, 2:4] = observed_data
        train_data[i, j, 1:34, 2:4] = observed_data

        # Initial condition (observed state one day before forecast)
        ic_data = observed_raw_data.sel(time=dates[a+j-1])
        test_data[j, 0, 0:2] = ic_data
        train_data[i, j, 0, 0:2] = ic_data
        test_data[j, 0, 6] = amp(ic_data[0], ic_data[1])
        train_data[i, j, 0, 6] = amp(ic_data[0], ic_data[1])

    a += size

    # Add lead time feature
    test1_data = np.concatenate((test_data, test_extra_column), axis=2)
    test2_data = np.zeros((366, 33, 34, 10))

    # Reshape to create samples for each forecast lead (0-32 days)
    for l in range(33):
        test2_data[:, l, 0:(l+2), :] = test1_data[:, 0:(l+2), :]

    # Remove samples with missing data
    mask1 = ~np.isnan(test2_data).any(axis=(2, 3))
    data_1 = test2_data[mask1]
    target_year = str(1981 + i)

    # Save test data for this year
    file_path_ec = f'../preprocessing_data/target_y{target_year}.test'
    np.save(file_path_ec, data_1)

# Create training and validation datasets (1996-2010)
# Use past 15 years of data for training each target year
for i in range(15):
    year = i + 15
    target_year = str(1981 + year)

    # Select 15 years of historical data
    selected1 = train_data[i:year, :, :, :]
    train1_data = np.concatenate((selected1, train_extra_column), axis=3)
    train2_data = np.zeros((15, 366, 33, 34, 10))

    # Reshape to create samples for each forecast lead
    for j in range(33):
        train2_data[:, :, j, 0:(j+2), :] = train1_data[:, :, 0:(j+2), :]

    train2_data = train2_data.reshape(15*366, 33, 34, 10)

    # Remove samples with missing data
    mask1 = ~np.isnan(train2_data).any(axis=(1, 2, 3))
    data1 = train2_data[mask1]

    # Split into training and validation sets
    indices = np.arange(540)
    selected_indices = np.linspace(0, 540 - 1, 150, dtype=int)  # 150 validation samples
    remaining_indices = np.delete(indices, selected_indices)
    data_train = data1[remaining_indices]  # 390 training samples
    data_validate = data1[selected_indices]

    # Save training and validation data
    np.save(f'../preprocessing_data/target_y{target_year}.train', data_train)
    np.save(f'../preprocessing_data/target_y{target_year}.validate', data_validate)

print('Data preprocessing complete')
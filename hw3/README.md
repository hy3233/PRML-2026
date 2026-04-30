# PRML 2026 - Assignment 3: Air Quality Forecasting

This assignment uses the Beijing Air Quality dataset to forecast the next-hour
PM2.5 concentration from the previous 24 hours of pollution and weather data.

## Files

```text
hw3/
  main.py                         # experiment code
  archive/
    LSTM-Multivariate_pollution.csv
    pollution_test_data1.csv
  figures/                        # generated plots
  results/                        # generated metrics
  report3.pdf                     # final report
```

## Run

```powershell
cd hw3
python main.py
```

Required Python packages:

```powershell
pip install numpy pandas matplotlib scikit-learn torch
```

## Method

The task is converted into supervised learning by flattening the previous
24 hourly observations into one feature vector. Numeric variables are
standardized using only the training period, while wind direction is one-hot
encoded. Models are evaluated on a chronological 80/20 split.

Compared models:

- Persistence baseline
- Ridge regression
- Random Forest
- Histogram Gradient Boosting
- Multilayer Perceptron
- PyTorch LSTM

The generated figures and metrics summarize the dataset, preprocessing, model
comparison, and conclusions.

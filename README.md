
# MMA3001 - Combined Cycle Power Plant Output Prediction

## 1. Project Overview

This project investigates the prediction of net electrical power output from a Combined Cycle Power Plant (CCPP) using regression models.

The objective is to predict electrical power output (PE) based on four environmental and operating variables.

Three prediction approaches are evaluated:
- Mean Baseline
- Linear Regression
- Decision Tree Regression

The project compares prediction accuracy, model complexity, computational performance, and limitations.

## 2. Dataset

Source: UCI Machine Learning Repository - Combined Cycle Power Plant

Original file: `data/Folds5x2_pp.xlsx`

Input variables:
- AT: Ambient Temperature (°C)
- V: Exhaust Vacuum (cm Hg)
- AP: Ambient Pressure (mbar)
- RH: Relative Humidity (%)

Output variable:
- PE: Net Electrical Power Output (MW)

The original dataset contains 9,568 observations.

After removing 41 exact duplicate rows, 9,527 observations remain.

The cleaned data is divided into:
- Training set: 6,668 samples
- Validation set: 1,429 samples
- Test set: 1,430 samples

A fixed random state of 42 is used for reproducibility.

## 3. Methodology

The main steps are:

1. Explore the dataset and examine relationships between variables.
2. Remove duplicate observations.
3. Divide the data into training, validation, and test sets.
4. Establish a mean prediction baseline.
5. Train a Linear Regression model.
6. Train a Decision Tree Regression model.
7. Select the decision tree depth using validation results.
8. Perform sensitivity and computational performance analyses.
9. Evaluate the selected models on the independent test set.

The selected decision tree uses a maximum depth of 8.

## 4. Final Test Results

| Model | Test MAE (MW) | Test RMSE (MW) | Test R2 |
|---|---:|---:|---:|
| Mean Baseline | 14.880 | 17.153 | ~0 |
| Linear Regression | 3.756 | 4.744 | 0.924 |
| Decision Tree (Depth 8) | 3.194 | 4.330 | 0.936 |

The Decision Tree achieved the lowest prediction errors on the test set.

Compared with Linear Regression, it reduced test MAE by approximately 14.96%.

## 5. Repository Structure

- `data/`: Original dataset and dataset description.
- `notebooks/`: Data exploration, regression models, validation, sensitivity analysis, and final testing.
- `src/`: Reusable Python functions for data preparation, model training, and evaluation.
- `tests/`: Unit tests and PyTest execution report.
- `docs/`: Generated HTML documentation.
- `README.md`: Project overview and instructions.

## 6. Software Requirements

The project uses Python and the following libraries:

- NumPy
- pandas
- Matplotlib
- scikit-learn
- openpyxl
- pytest
- pdoc

The Jupyter notebooks were developed using Google Colab.

## 7. How to Run the Project

### Jupyter Notebooks

1. Download or clone this repository.
2. Open the required notebook in Google Colab.
3. Upload `Folds5x2_pp.xlsx` to the Colab working directory.
4. Run the notebook cells from top to bottom.

The notebooks are numbered in the intended analysis order.

### Unit Tests

From the repository root directory, run:

    python -m pytest tests -v

Four unit tests check data loading, data splitting, model training, and evaluation metrics.

The test execution report is available at:

`tests/test_report.txt`

### HTML Documentation

The generated API documentation is stored in:

`docs/index.html`

The documentation describes the functions available in `src/power_plant.py`.

## 8. Engineering Interpretation and Limitations

The model may support preliminary power output estimation and analysis under operating conditions represented by the dataset.

However, the model has limitations:

- Only four input variables are considered.
- Some observations produce relatively large prediction errors.
- The maximum absolute test prediction error was 49.895 MW.
- The evaluation uses random partitions of the same dataset.
- Performance on different power plants or future operating conditions has not been independently verified.

The model should not be treated as a validated real-time power plant control system.

## 9. AI Assistance

ChatGPT was used to assist with project planning, explanations, code and documentation drafting, and troubleshooting.

The computational results were checked through model evaluation, visualisations, and unit tests.

Further discussion of AI use, verification, and limitations is provided in the project report.

## 10. Data Attribution

Dataset: Combined Cycle Power Plant

Source: UCI Machine Learning Repository

The original dataset is attributed to its respective creators. The licensing of project code does not alter the original dataset's terms.

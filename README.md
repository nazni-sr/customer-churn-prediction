# Customer Attrition Prediction

Machine learning project predicting customer churn, built as a portfolio project for Data Analyst / BI Analyst / Entry-Level ML roles.

**Status:** Data cleaning and exploratory analysis complete. Feature engineering and modeling in progress.

## Overview

This section will describe the business problem, dataset, modeling approach, and results once the project is built out.

## Dataset

**[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)** (originally released by IBM as a sample dataset for customer retention analysis).

- 7,043 customers, 21 features (demographics, account info, services subscribed)
- Target: `Churn` (Yes/No) — 26.5% churn rate (realistic class imbalance)
- Business context: a telecom provider's customer-level data, used to predict which customers are likely to cancel their subscription

The raw CSV is not committed to this repo (data files shouldn't live in git history). To reproduce:

```bash
# Download into data/raw/
curl -o data/raw/Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

## Exploratory Data Analysis

Full analysis: [`notebooks/01_data_cleaning_and_eda.ipynb`](notebooks/01_data_cleaning_and_eda.ipynb) · Business writeup: [`reports/eda_summary.md`](reports/eda_summary.md)

Key finding — contract type is the strongest churn driver found so far:

![Churn rate by contract type](reports/figures/churn_by_contract.png)

Month-to-month customers churn at **42.7%**, vs. **11.3%** for one-year and **2.8%** for two-year contracts. See the full writeup for all findings (tenure, pricing, internet service, and payment method effects).

## Tech Stack

- Python
- scikit-learn
- SHAP
- Streamlit

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/          # Original, unmodified datasets (not committed)
│   └── processed/    # Cleaned/engineered data ready for modeling
├── notebooks/         # Exploratory analysis (EDA, prototyping)
├── src/                # Reusable, production-quality Python modules
├── models/            # Saved trained model artifacts (not committed)
├── app/                # Streamlit application
├── tests/              # Unit tests
├── reports/
│   └── figures/       # Saved plots/visualizations for the README and reports
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# Clone the repo
git clone https://github.com/nazni-sr/customer-churn-prediction.git
cd customer-churn-prediction

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## License

See [LICENSE](LICENSE).

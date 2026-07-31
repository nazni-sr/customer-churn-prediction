# Customer Attrition Prediction

Machine learning project predicting customer churn, built as a portfolio project for Data Analyst / BI Analyst / Entry-Level ML roles.

**Status:** Project setup in progress — dataset selection, EDA, modeling, and deployment sections coming soon.

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

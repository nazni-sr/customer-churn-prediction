# Customer Attrition Prediction

Machine learning project predicting customer churn, built as a portfolio project for Data Analyst / BI Analyst / Entry-Level ML roles.

**Status:** Project setup in progress — dataset selection, EDA, modeling, and deployment sections coming soon.

## Overview

This section will describe the business problem, dataset, modeling approach, and results once the project is built out.

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

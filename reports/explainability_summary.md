# Explainability Summary — Business Findings

The model's predictions were explained using SHAP (SHapley Additive exPlanations), applied to the tuned Random Forest selected in the modeling stage. SHAP assigns every feature a contribution value for every individual prediction, which enables two things a plain accuracy score cannot: (1) confidence that the model is learning real patterns rather than noise, and (2) a concrete, per-customer explanation a retention team can act on.

## Global drivers (confirmed two independent ways)

The top churn drivers identified by SHAP match the mutual information ranking from the feature selection stage almost exactly:

1. **Internet service type** — fiber optic customers are pushed toward churn; DSL or no internet pushes away from it
2. **Tenure** — shorter tenure consistently pushes toward churn
3. **Contract type** — two-year contracts strongly push away from churn; month-to-month pushes toward it
4. **Payment method** — electronic check pushes toward churn; other payment methods do not
5. **Monthly charges** — higher charges push toward churn

Two independent methods (a statistical test and a trained model's internal logic) converging on the same top factors is meaningfully more trustworthy than either one alone — it substantially lowers the chance these are spurious correlations.

## What this enables that a plain probability score doesn't

A churn probability alone tells a retention team *who* to worry about, not *why* — which makes it hard to design an effective intervention. SHAP closes that gap:

- **High-risk customer example**: flagged at 96.4% churn probability, explained by short tenure, fiber optic service, new-customer status, no long-term contract, and electronic check payment. A retention agent reading this knows the highest-leverage offer is probably a contract upgrade incentive, not a generic discount.
- **Model limitation, disclosed honestly**: one customer with a low predicted risk (21.4%) churned anyway. Nearly every available feature pointed toward retention for this customer — the model's explanation shows *why* it was confident, and that confidence was reasonable given the data available, even though the outcome differed. This is a real limitation of any model built on account-level data alone: it can't see service complaints, competitor offers, or other unrecorded events that also drive churn.

## Recommendation

Any deployment of this model (see the Streamlit app) should surface the top 3–5 SHAP-driven reasons alongside each risk score, not the score in isolation. A number without a reason is not actionable for a frontline retention team; a number with a reason is.

*Full analysis: `notebooks/04_explainability.ipynb`. Supporting figures: `reports/figures/shap_*.png`.*

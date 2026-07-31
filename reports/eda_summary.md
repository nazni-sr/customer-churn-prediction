# EDA Summary — Business Findings

**Dataset:** Telco Customer Churn, 7,043 customers. **Overall churn rate: 26.5%.**

## What drives churn

| Factor | Finding | Business read |
|---|---|---|
| **Contract type** | Month-to-month: 42.7% churn · One-year: 11.3% · Two-year: 2.8% | Contract length is the single strongest lever available. Customers with no cancellation penalty leave at 15x the rate of two-year customers. |
| **Tenure** | Churned customers: 10 months median · Retained: 38 months median | Churn risk is front-loaded — the first year of a customer's lifecycle is the highest-risk window. |
| **Monthly charges** | Churned: $79.65 median · Retained: $64.43 median | Customers who churn are paying more, not less — suggests price sensitivity or a mismatch between price and perceived value. |
| **Internet service** | Fiber optic: 41.9% churn · DSL: 19.0% · No internet: 7.4% | Fiber optic customers churn more than double the rate of DSL customers — worth investigating service quality or pricing complaints in that segment. |
| **Payment method** | Electronic check: 45.3% churn · Other methods: 15–19% | Electronic check users churn at more than double any other payment method. Automatic payments (bank transfer, credit card) correlate with the lowest churn — likely a proxy for lower-friction, more committed customers. |

## Preliminary business recommendations

1. **Incentivize longer contracts.** Even modest discounts for switching month-to-month customers to annual contracts could meaningfully cut churn, given the 15x gap in churn rate by contract type.
2. **Target retention efforts in the first year.** Onboarding and early-tenure engagement (first 10–12 months) is where churn risk concentrates — this is where a proactive retention program has the most leverage.
3. **Investigate the fiber optic and electronic-check segments specifically.** Both show outsized churn independent of contract length, suggesting service- or billing-experience issues rather than just price.
4. **Promote automatic payment enrollment.** The correlation with lower churn is strong enough to justify a targeted campaign, though causation should be tested rather than assumed (see caveat below).

## Caveats

- These are **univariate associations**, not causal effects — e.g., payment method may be a proxy for customer type rather than a direct churn driver. Confirming which features are actually predictive (and by how much) is the job of the modeling and explainability stages, not EDA.
- Class imbalance (26.5% positive rate) will need explicit handling in modeling (class weights or resampling), not just accuracy-based evaluation.

*Full analysis: `notebooks/01_data_cleaning_and_eda.ipynb`. Supporting figures: `reports/figures/`.*

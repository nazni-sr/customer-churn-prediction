# Modeling Results — Business Summary

Three models were trained, tuned, and compared: Logistic Regression, Random Forest, and Gradient Boosting. All three were evaluated on a held-out test set (1,409 customers) never used during training or tuning.

## Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (selected)** | 0.756 | 0.527 | 0.791 | 0.632 | **0.846** |
| Logistic Regression | 0.740 | 0.507 | 0.805 | 0.622 | 0.844 |
| Gradient Boosting | 0.749 | 0.517 | 0.799 | 0.628 | 0.842 |

**All three models perform within 0.4 points of ROC-AUC of each other.** This matters: it means the choice of algorithm mattered far less than the feature engineering work in the previous stage. A simple, highly interpretable Logistic Regression captures nearly as much predictive signal as more complex tree ensembles — the churn signal in this data is largely explained by a handful of strong, additive factors (contract type, tenure, service type, payment method) rather than complex interactions between features.

## What the numbers mean for the business

The selected model (Random Forest) correctly identifies **79.1% of customers who actually churn** (296 of 374 in the test set). It also flags 266 customers who ultimately would not have churned (false positives).

**Why this trade-off was chosen deliberately:** in a retention context, missing a customer who was about to leave (a false negative) is far more costly than offering an unnecessary retention incentive to someone who wasn't going to churn (a false positive) — a missed churner is a customer lost outright, while a false positive costs a discount or outreach call. The model was tuned to favor catching churners over minimizing false alarms, which is the correct default for a retention use case.

## Recommendations

1. **Deploy the Random Forest model to flag at-risk customers**, prioritizing customers with month-to-month contracts, fiber optic internet, electronic check payment, and short tenure — the strongest predictors identified across EDA, feature selection, and modeling.
2. **Tune the decision threshold against real campaign economics**, not the default 0.5 cutoff. If a retention offer costs far less than the revenue lost from a churned customer, the threshold should be pushed even further toward recall.
3. **Treat this as a starting point, not a finished system.** The next stage (explainability) will clarify *why* the model flags a given customer, which is necessary before frontline retention teams can act on these predictions with confidence.

## Caveats

- Metrics reflect a single train/test split; cross-validation scores (reported in the notebook) confirm the ranking is stable, but exact numbers would shift slightly with a different random seed.
- The model was not validated against a temporally held-out period (e.g., a later cohort of customers) — this is a static snapshot, not a monitored production model. Performance should be re-evaluated periodically if deployed.

*Full analysis: `notebooks/03_modeling.ipynb`. Supporting figures: `reports/figures/`. Raw comparison table: `reports/model_comparison.csv`.*

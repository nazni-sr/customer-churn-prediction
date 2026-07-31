# Resume Bullets & Interview Prep

Grounded in the actual project — every number here is real, pulled from `reports/model_results.md`, `reports/explainability_summary.md`, and the git history, not filler.

## Resume Bullets (ATS-friendly)

Pick 3–5 depending on the role. The first three lean technical (ML/Data Analyst roles); the last three lean business/communication (BI/Business Analyst roles).

- Built an end-to-end customer churn prediction pipeline in Python (pandas, scikit-learn) on a 7,043-customer dataset, covering data cleaning, EDA, feature engineering, model training, and deployment, version-controlled across 10+ incremental Git commits.
- Engineered and statistically validated 5 candidate features via mutual information analysis; identified and removed 9 redundant/multicollinear features from one-hot encoding, reducing model input dimensionality by 24% with no loss in predictive signal.
- Trained and hyperparameter-tuned 3 classification models (Logistic Regression, Random Forest, Gradient Boosting) via 5-fold cross-validated grid search; selected the final model by ROC-AUC (0.846) and recall (79%) rather than raw accuracy, appropriate for a 26.5% class-imbalance problem.
- Applied SHAP to explain both global and individual-customer churn predictions, cross-validating model-driven feature importance against an independent statistical method and translating findings into specific retention recommendations (e.g., contract-length incentives, fiber-optic service review).
- Deployed an interactive Streamlit web application allowing non-technical users to score a customer's churn risk and see the top contributing factors in plain language, not just a probability.
- Diagnosed and fixed a silent one-hot-encoding bug in single-record inference that would have produced systematically incorrect predictions in production, catching it through deliberate before/after verification rather than assuming code correctness from a lack of errors.

**Project line for a resume header/summary:**
`Customer Churn Prediction System — Python, scikit-learn, SHAP, Streamlit | github.com/nazni-sr/customer-churn-prediction`

## Interview Explanation

### The 60–90 second pitch (answer to "tell me about a project")

"I built an end-to-end churn prediction system using the Telco Customer Churn dataset — about 7,000 telecom customers, 26.5% of whom churned. The business problem is straightforward: it's much cheaper to retain a customer than acquire a new one, so the goal was to flag at-risk customers *before* they cancel, with enough explanation that a retention team could actually act on it, not just a black-box score.

I went through the full pipeline: cleaned the data, did EDA to find the strongest churn signals — contract type turned out to be huge, month-to-month customers churn at 15 times the rate of two-year contract customers — then engineered and validated new features, dropping ones that turned out to be redundant. I trained and tuned three models — logistic regression, random forest, and gradient boosting — and picked the random forest based on ROC-AUC and recall, since missing an actual churner is more costly than a false alarm. Then I used SHAP to explain individual predictions, and wrapped the whole thing in a Streamlit app where you can enter a customer's details and get both a risk score and the reasons behind it."

### Business problem

- Churn directly costs recurring revenue; acquiring a replacement customer is typically far more expensive than retaining an existing one.
- The deliverable isn't "a model" — it's a decision-support tool. A probability alone isn't actionable for a retention team; they need to know *why* a customer is at risk to design the right intervention.

### Technical approach

- Structured the codebase like production software, not a single notebook: reusable modules in `src/` (`data_cleaning.py`, `feature_engineering.py`, `feature_selection.py`, `model_pipeline.py`, `train.py`, `explainability.py`, `inference.py`), with notebooks used only for exploration and orchestration on top of that tested code.
- Handled a real data quality issue: `TotalCharges` loaded as text because new customers (tenure = 0) had blank values instead of 0 — required explicit type coercion and a deliberate fill-with-zero decision, not just `dropna()`.
- Engineered features with hypotheses, then validated them statistically (mutual information) rather than assuming they'd help — one feature (`avg_monthly_spend`) was built, tested, and dropped after turning out 99.6% correlated with an existing column.
- Used stratified train/test splitting and 5-fold cross-validation throughout; the test set was touched exactly once, only for final evaluation, to keep the reported metrics honest.

### Model choice

- Compared three model families — logistic regression (interpretable baseline), random forest, and gradient boosting — rather than jumping straight to the most complex option.
- **All three converged to within 0.4 points of ROC-AUC after tuning.** That's a real finding worth stating directly in an interview: it means the feature engineering mattered more than the algorithm choice here, because the churn signal in this data is largely additive rather than driven by complex feature interactions. Be ready to explain *why* that's plausible (a handful of strong, independent factors — contract, tenure, service type, payment method — rather than subtle combinations of many weak signals).
- Selected Random Forest by test-set ROC-AUC, but would be equally comfortable defending Logistic Regression as the production choice in a real setting, given its near-identical performance and much higher interpretability — a good example of knowing that "best score" and "best choice for the business" aren't always the same thing.

### Evaluation

- Used ROC-AUC as the primary metric (not accuracy) because of the 26.5% class imbalance — a model that predicts "no churn" for everyone would hit ~73.5% accuracy while catching zero actual churners.
- Applied `class_weight="balanced"` on every model to prevent exactly that failure mode.
- Reported the full metric suite (accuracy, precision, recall, F1, ROC-AUC, confusion matrix) rather than cherry-picking the best-looking one, and explicitly explained the precision/recall trade-off: the selected model catches 79.1% of churners at the cost of a 47% false-positive rate among flagged customers — a deliberate choice, since a missed churner is a lost customer while a false positive just costs an unnecessary retention offer.

### Business impact

- Translated every technical finding into a business sentence, not just a chart: e.g. "month-to-month customers churn at 42.7% vs. 2.8% for two-year contracts" → contract-length incentives are the highest-leverage retention lever available.
- Identified that fiber optic customers and electronic-check payers churn at roughly double the rate of other segments independent of contract length — flagged as worth investigating for service-quality or billing-experience issues specifically, not just addressed with a blanket retention offer.
- Explicitly noted the model's limits rather than overselling it: a false-negative example (predicted 21% risk, churned anyway) showed the model can't see things like a bad support interaction or a competitor's offer — real churn drivers outside the available data.

### Questions to be ready for

- **"Why didn't you just use the model with the highest score?"** — Because all three were statistically indistinguishable; algorithm choice mattered less than feature quality, and in a real deployment I'd weigh interpretability and maintenance cost, not just the third decimal place of ROC-AUC.
- **"How do you know the model isn't overfitting?"** — Cross-validation scores on the training set were consistent with the held-out test set performance (both around 0.84–0.85 ROC-AUC), and the test set was never used for any model-selection decision, only final evaluation.
- **"What would you do differently, or next?"** — Tune the classification threshold against actual retention-offer economics instead of the default 0.5 cutoff; validate on a temporally later cohort rather than a single random split, since customer behavior can drift over time; consider monitoring for model drift if this were actually deployed on an ongoing basis.
- **"Tell me about a bug you had to debug."** — The one-hot encoding issue is the strongest answer available: `pd.get_dummies()` on a single customer row only creates a column for whichever category is present in that one row, so a customer with fiber optic internet silently got a `0` for `InternetService_Fiber optic` instead of `1` — no error, just a wrong prediction. Caught it by explicitly testing the encoded output against known input before trusting the model's prediction, not by assuming correctness because the code ran without errors.

# 🏦 Bank Churn Forensic Audit: A High-Recall Risk Assessment Tool

This project moves beyond standard churn prediction to perform a Forensic Audit of bank customer behavior. By synthesizing raw data into high-impact engineered features, this analysis identifies the "Underutilized Elite"—a segment of high-value customers hiding in plain sight who pose the greatest flight risk.


## 📊 Executive Highlights & Methodology

### 1. Data Sanitization & Prep
   
    1.1 Noise Reduction: Dropped non-predictive identifiers (RowNumber, CustomerId, Surname) to prevent model overfitting and dimensionality bloat.

    1.2 Categorical Translation: Applied One-Hot Encoding (drop_first=True) to Geography and Gender to eliminate the dummy variable trap (multicollinearity).

    1.3 Imbalance Handling: Utilized stratify=y during the 80/20 train-test split to ensure the model learns from a mathematically accurate representation of the minority "churn" class.

    1.4 Normalization: Applied StandardScaler to ensure high-magnitude continuous variables (like Balance) do not overpower low-magnitude signals (like NumOfProducts).

### 2. Forensic EDA (Identifying the Leaks)
   
    2.1 The Age Flaw: Pinpointed the 40–50 age bracket as the primary flight risk.

    2.2 The Capital & Engagement Paradox: Discovered the "Zero-Balance Zombie" risk (52% are active but hold no funds, providing zero switching friction) and the "High-Value Leak" (inactive users sitting on $100k+).

    2.3 The Subprime & Legacy Exits: Identified through outlier analysis that extreme low-credit users are being flushed out, while legacy users aged 75+ essentially never churn.

### 3. The Whale Audit & Product Paradox
   
    3.1 The German Multiplier: Proved that high-balance German customers are 2.03x more likely to churn than their French/Spanish counterparts.

    3.2 The Cross-Sell Failure: Uncovered that holding 3+ products is effectively an "Auto-Churn" trigger (95–100% exit rate), suggesting toxic bundles or restrictive fee structures for high-net-worth clients.

### 4. Feature Engineering & Statistical Proof
   
      4.1 Engineered Signals: Synthesized complex user behaviors into mathematically pure flags (Balance_Salary_Ratio, Tenure_By_Age, Is_Inactive_Whale, Credit_Score_Per_Product).
   
      4.2 Statistical Validation: Ran Chi-Square testing to prove the German regional risk and the Inactive Whale flag are mathematically undeniable realities ($p < 0.001$), not random noise.


### 5. Predictive Modeling & Calibration
   
     5.1 The Baseline Failure: A standard model achieved 81% accuracy but suffered from "Accuracy Illusion"—missing 81% of actual churners (19% Recall).
   
     5.2 The Engineering Breakthrough: Injecting engineered features nearly doubled the raw detection rate to 32%. Credit_Score_Per_Product emerged as the #1 predictive driver (Weight: +3.24).
   
     5.3 Stability Proof (K-Fold): A 5-Fold Cross-Validation yielded a highly stable Mean Weight of 3.2437 ($\sigma = 0.0473$), proving the signal is a "Global Truth."
     5.4 The Balanced Calibration: Applied class_weight='balanced' to shift the decision boundary, successfully pushing final Recall to 74%.

## 🎯 Strategic Intervention Plan

     Based on the extracted model coefficients, we mapped mathematical drivers to actionable business strategies to safeguard revenue.

| Segment | Forensic Driver (Weight) | Intervention Strategy | Priority |
| :--- | :--- | :--- | :--- |
| **The Underutilized Elite** | `CS_Per_Product` (+3.24) | **"The Anchor Offer":** Target customers with High Credit Scores and only 1 product. Offer a "Premium Multi-Product Bundle" to increase switching friction. | **CRITICAL** |
| **The Inactive Whales** | `IsActiveMember` (-0.54) | **"Concierge Re-engagement":** Flag high-balance members who haven't logged in for 90 days. Trigger a "Portfolio Health Check" outreach call. | **HIGH** |
| **The German Leak** | `Geography_Germany` (+0.41) | **"Regional Competitive Audit":** Investigate local German competitors. Is a specific neo-bank offering better interest rates or a superior UI/UX? | **MEDIUM** |
| **The Vulnerable Seniors** | `Age` (+0.56) | **"Legacy Loyalty Program":** Implement a "Senior Priority" tier with better in-person service or estate planning perks to reward tenure. | **MEDIUM** |

## 🧠 The Math Behind the Machine: Logistic Regression

     Unlike "Black Box" algorithms, Logistic Regression was chosen for its strict mathematical interpretability, allowing us to reverse-engineer the exact drivers of churn.

  ### The Core Equation

     The model calculates the probability $P$ that a customer will churn ($y=1$) given their specific features $\mathbf{X}$, using the Sigmoid Function:
      $$P(y=1|\mathbf{X}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n)}}$$
  
  
### Inner Workings

      1. The Linear Combination: The model calculates a raw log-odds score by multiplying each feature ($X_n$) by a learned weight ($\beta$).A positive weight (e.g., Age) pushes the score higher, driving the prediction toward Churn.
     2. A negative weight (e.g., IsActiveMember) pulls the score lower, indicating stability.The Sigmoid Transformation: This raw score is squashed into a strict probability between 0 and 1.
     3. The Decision Boundary: By utilizing class_weight='balanced', we mathematically lowered the probability threshold required to flag a "Churner," prioritizing the capture of at-risk revenue over raw precision.


## ⚠️ Limitations & The Way Forward

      Limitations
      1. Linearity Constraint: Logistic Regression assumes a linear relationship. It may miss complex, multi-dimensional interactions without manual feature engineering.

      2. Precision-Recall Trade-off: To achieve 74% Recall, we accept a Precision of 42% (False Alarms). This is optimal for low-cost outreach but less so for high-cost bonuses.

## The Way Forward
      1. Ensemble Expansion: Implement XGBoost or LightGBM to capture non-linear interactions and potentially increase the F1-Score.
      2. A/B Testing: Launch a pilot of the "Anchor Offer" to a control group of the "Underutilized Elite" to measure actual churn reduction.
      3. Temporal Integration: Incorporate time-series data (e.g., "Days since last transaction") to transform this from a static audit into a real-time early warning system.
   









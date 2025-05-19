# German Credit Risk & Fairness Analysis 🧠📊

🤖 Automated credit scoring using H2O AutoML with fairness evaluation by gender.  
Includes data preprocessing, SHAP explainability, parity metrics, and visual risk profiling.  
The project is fully written in R and structured for reproducibility.

---

## 📁 Project Structure

- `data/` – raw input data (`.csv`)
- `output/` – generated outputs (plots, predictions)
- `src/` – core analysis script
- `README.md` – project documentation

---

## 🔢 Block 1: Load required packages and get script path

📄 [`src/analysis.R`](src/analysis.R)  
This block loads required libraries and dynamically detects the working directory of the script to ensure all relative paths work properly.

```r
if (!require("rstudioapi")) install.packages("rstudioapi")
library(rstudioapi)

script.dir <- dirname(rstudioapi::getSourceEditorContext()$path)

📄 src/analysis.R
📥 german_credit_data_small.csv
This block loads the CSV file containing credit data and drops the unnecessary ID column.

data <- read.csv(file.path(script.dir, "german_credit_data_small.csv"))
data <- data[, 2:11]  # Drop ID column
📝 The dataset contains information on credit applicants, including age, sex, job type, housing status, account balances, loan amount, duration, purpose, and risk classification.
Removing the ID column helps ensure that modeling is not biased by an arbitrary identifier.

📄 src/analysis.R
This block transforms categorical variables into numerical or factor encodings to prepare them for modeling.

library(dplyr)

data$Sex <- as.factor(recode(data$Sex, 'male' = 1, 'female' = 2))
data$Housing <- as.factor(recode(data$Housing, 'free' = 1, 'rent' = 2, 'own' = 3))
...
data$Risk <- as.factor(recode(data$Risk, 'good' = 0, 'bad' = 1))

🎯 Variables like Sex, Housing, Purpose, Saving.accounts, and Risk are transformed into factors with consistent numeric codes.
This ensures compatibility with machine learning algorithms and fairness analysis later in the script.


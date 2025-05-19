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
```

✅ This makes the script self-contained and portable — no hardcoded file paths required.


## 🔢 Block 2: Load and prepare the dataset

📄 [`src/analysis.R`](src/analysis.R)  
📥 [`german_credit_data_small.csv`](data/german_credit_data_small.csv)  
This block loads the CSV file containing credit data and drops the unnecessary ID column.

```r
data <- read.csv(file.path(script.dir, "german_credit_data_small.csv"))
data <- data[, 2:11]  # Drop ID column
```

📝 The dataset contains information on credit applicants, including age, sex, job type, housing status, account balances, loan amount, duration, purpose, and risk classification.
Removing the ID column helps ensure that modeling is not biased by an arbitrary identifier.


## 🔢 Block 3: Recode categorical variables

📄 [`src/analysis.R`](src/analysis.R)  
This block transforms categorical variables into numerical or factor encodings to prepare them for modeling.

```r
library(dplyr)

data$Sex <- as.factor(recode(data$Sex, 'male' = 1, 'female' = 2))
data$Job <- as.factor(data$Job)
data$Housing <- as.factor(recode(data$Housing, 'free' = 1, 'rent' = 2, 'own' = 3))
data$Saving.accounts <- as.factor(recode(data$Saving.accounts, 'little' = 1, 'moderate' = 2, 
                                         'quite rich' = 3, 'rich' = 4))
data$Checking.account <- as.factor(recode(data$Checking.account, 'little' = 1, 'moderate' = 2))
data$Credit.amount <- as.numeric(data$Credit.amount)
data$Duration <- as.numeric(data$Duration)
data$Purpose <- as.factor(recode(data$Purpose, 'vacation/others' = 1, 'radio/TV' = 2, 
                                 'domestic appliances' = 3, 'furniture/equipment' = 4,
                                 'repairs' = 5, 'car' = 6, 'education' = 7, 'business' = 8))
data$Risk <- as.factor(recode(data$Risk, 'good' = 0, 'bad' = 1))
```

🎯 Variables like Sex, Housing, Purpose, Saving.accounts, and Risk are transformed into factors with consistent numeric codes.
This ensures compatibility with machine learning algorithms and fairness analysis later in the script.


## 🔢 Block 4: Handle missing values

📄 [`src/analysis.R`](src/analysis.R)  
This block checks for missing values and imputes them using the MICE (Multivariate Imputation by Chained Equations) method.

```r
library(mice)

# Check proportion of missing values
sum(is.na(data)) / (nrow(data) * ncol(data))

# Perform imputation
data <- complete(mice(data))

# Confirm that missing values are handled
sum(is.na(data)) / (nrow(data) * ncol(data))
```

📊 The script calculates the percentage of missing data and uses iterative chained equations to impute values.
This helps maintain statistical integrity and prevents bias due to incomplete records.


## 🔢 Block 5: Create and save scatter plot

📄 [`src/analysis.R`](src/analysis.R)  
📈 [`dotplot_germancredit.png`](output/dotplot_germancredit.png)  
This block creates a scatter plot visualizing the relationship between credit amount and duration, colored by credit risk and shaped by gender.

```r
library(ggplot2)

ggplot(data = data) +
  geom_point(aes(x = Duration, y = Credit.amount, color = as.factor(Risk), shape = as.factor(Sex)), size = 2) +
  theme(text = element_text(size = 15))

ggsave(filename = file.path(script.dir, "dotplot_germancredit.png"), width = 5, height = 4.5, dpi = 100)
```

🖼️ The plot helps visually identify how loan characteristics relate to credit risk and gender, highlighting potential biases or patterns in approvals.


## 🔢 Block 6: Export processed dataset

📄 [`src/analysis.R`](src/analysis.R)  
📤 [`german_credit_data_small_processed.csv`](data/german_credit_data_small_processed.csv)  
This block saves the cleaned and transformed dataset to a new CSV file for later modeling steps.

```r
write.csv(data, file.path(script.dir, "german_credit_data_small_processed.csv"))
```

💾 This file serves as the final input for the AutoML pipeline.
Exporting it separately ensures a clean separation between data preprocessing and model training logic.


## 🔢 Block 7: Start H2O and load data

📄 [`src/analysis.R`](src/analysis.R)  
This block initializes the H2O framework, converts the dataset into an H2O frame, and splits it into training and testing subsets.

```r
library(h2o)
h2o.init()

# Convert data to H2O frame
frame <- as.h2o(data)

# Split into training (80%) and test (20%) sets
splits <- h2o.splitFrame(frame, ratios = 0.8)
train <- splits[[1]]
test <- splits[[2]]
```

🚀 The model will be trained on 80% of the data and validated on the remaining 20%.
This split is essential for evaluating how well the model generalizes to unseen data.


## 🔢 Block 8: Train model using H2O AutoML

📄 [`src/analysis.R`](src/analysis.R)  
This block runs automated machine learning (AutoML) using H2O, training multiple models and selecting the best one based on performance.

```r
y <- "Risk"

aml <- h2o.automl(
  y = y,
  training_frame = train,
  max_runtime_secs = 120,
  seed = 123,
  exclude_algos = "StackedEnsemble"
)
```

🧠 This greatly simplifies model selection and tuning by letting the AutoML engine explore different algorithms and hyperparameters.
The best-performing model is saved as aml@leader.


## 🔢 Block 9: Explain model behavior with SHAP

📄 [`src/analysis.R`](src/analysis.R)  
This block explains the output of the model using SHAP (SHapley Additive exPlanations) values, which reveal the contribution of each feature to individual predictions.

```r
exa <- h2o.explain(aml, test)
exm <- h2o.explain(aml@leader, test)

# SHAP explanation for a single observation
h2o.shap_explain_row_plot(aml@leader, test, row_index = 1)
```

🔍 SHAP values provide transparency by quantifying the effect each feature has on a prediction.
This is crucial in high-stakes areas like credit scoring, where model decisions must be explainable.


## 🔢 Block 10: Fairness analysis by gender

📄 [`src/analysis.R`](src/analysis.R)  
This final block evaluates the model's fairness using multiple bias criteria, comparing prediction outcomes between male and female applicants.

```r
# Generate predictions
test2 <- test
test2$predict <- h2o.predict(aml@leader, newdata = test)

# Criterion 1: Statistical (demographic) parity
mean(as.numeric(test2[test2$Sex == 1, ]$predict) - 1)  # Male rejection rate
mean(as.numeric(test2[test2$Sex == 2, ]$predict) - 1)  # Female rejection rate

# Criterion 2: Odds ratio
(
  mean(as.numeric(test2[test2$Sex == 1, ]$predict) - 1) /
  mean(as.numeric(test2[test2$Sex == 2, ]$predict) - 1)
)

# Criterion 3: Predictive parity (true positive rate)
mean(as.numeric(test2[test2$Sex == 1 & test2$Risk == 1, ]$predict) - 1)
mean(as.numeric(test2[test2$Sex == 2 & test2$Risk == 1, ]$predict) - 1)

# Criterion 4: Accuracy equality
mean(as.numeric(test2[test2$Sex == 1, ]$Risk == test2[test2$Sex == 1, ]$predict))
mean(as.numeric(test2[test2$Sex == 2, ]$Risk == test2[test2$Sex == 2, ]$predict))
```

⚖️ Fairness criteria computed:

Statistical (demographic) parity – equal rejection rates across genders

Odds ratio – ratio of rejection probabilities

Predictive parity – equal true positive rates

Accuracy equality – equal prediction accuracy between groups

✅ This analysis helps detect potential discrimination or unfair treatment based on gender.

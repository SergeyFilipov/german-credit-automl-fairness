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


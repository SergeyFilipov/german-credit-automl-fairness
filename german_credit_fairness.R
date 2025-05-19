if (!require("rstudioapi")) install.packages("rstudioapi")
library(rstudioapi)

# Get the path of the current script when using RStudio
script.dir <- dirname(rstudioapi::getSourceEditorContext()$path)

# Load the CSV file located in the same directory as the script
data <- read.csv(file.path(script.dir, "german_credit_data_small.csv"))


data <- data[,2:11] # Drop needless ID (var X)

# Recode All Variables to Numeric Values
library(dplyr)
data$Age <- as.numeric(data$Age)
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


# Impute and complete missing data
sum(is.na(data))/(dim(data)[1]*dim(data)[2]) # Check What Percentage are Missing

library(mice)
data <- complete(mice(data))
sum(is.na(data))/(dim(data)[1]*dim(data)[2]) # Check What Percentage are Missing


# Visualize gender and creditworthiness 
library(ggplot2)
ggplot(data = data) + geom_point(mapping = aes(x = Duration, y = Credit.amount, color = as.factor(Risk), 
                                               shape = as.factor(Sex)), size = 2) + theme(text = element_text(size=15))
# Define the directory of the current script
script.dir <- dirname(rstudioapi::getSourceEditorContext()$path)

# Save the graphic in the script directory
ggsave(filename = file.path(script.dir, "dotplot_germancredit.png"), 
       plot = last_plot(), width = 5, height = 4.5, dpi = 100)

dev.off()

# Export data set after processing
write.csv(data, "german_credit_data_small_processed.csv")

# Start ML Framework H2O and initiate connection to local host
library(h2o)
h2o.init()

# Import dataset

# frame <- h2o.importFile("german_credit_data_small_processed.csv")
frame <- as.h2o(data)


# Split into training and testing sub-sets
splits <- h2o.splitFrame(frame, ratios = 0.8)
train <- splits[[1]]
test <- splits[[2]]

y <- "Risk"

# Automatically Train AutoML models
aml <- h2o.automl(y = y, training_frame = train, max_runtime_secs = 120, 
                  seed = 123, exclude_algos = "StackedEnsemble")

exa <- h2o.explain(aml, test)
exm <- h2o.explain(aml@leader, test)

# Explain variable contributions to individual observations
h2o.shap_explain_row_plot(aml@leader, test, row_index = 1)

## Calculate probabilities as a proxy for fairness

# Create a dataset with actual values and model predictions
test2 <- test
test2$predict <- h2o.predict(aml@leader, newdata = test)

# Criterion 1: Statistical (or demographic) parity is achieved when the different groups (under the protected variable) have equal chance to be classified in a given class.
# If we have achieved statistical parity, then the two conditional probabilities would be equal.

#Calculate probabilities of getting rejected for credit for males (Sex = 1)
mean(as.numeric(test2[test2$Sex == 1,]$predict)-1)

#Calculate probabilities of getting rejected for credit for females (Sex = 2)
mean(as.numeric(test2[test2$Sex == 2,]$predict)-1)

# Criterion 2: Odds Ratio. Calculate odds ratio on protected variable: 1 means full equality, 0 mean full inequality
(mean(as.numeric(test2[test2$Sex == 1,]$predict)-1))/(mean(as.numeric(test2[test2$Sex == 2,]$predict)-1))

# Criterion 3: Predictive Parity is achieved if the two groups have equal chance of being classified in the same class, given that they belong to it

# Predict to reject a credit if a credit is actually bad for males
mean(as.numeric(test2[test2$Sex == 1 & test2$Risk == 1,]$predict)-1)

# Predict to reject a credit if a credit is actually bad for females
mean(as.numeric(test2[test2$Sex == 2 & test2$Risk == 1,]$predict)-1)

# If the two conditional probabilities are equal, then forecast parity holds.

# Criterion 4: Overall Accuracy Equality. This is achieved when the probability of the forecast to equal the actual class  does not change with respect to the protected variable

# Calculate probability of prediction to equal actuals for males
mean(as.numeric(test2[test2$Sex == 1,]$Risk == test2[test2$Sex == 1,]$predict))

# Calculate probability of prediction to equal actuals for females
mean(as.numeric(test2[test2$Sex == 2,]$Risk == test2[test2$Sex == 2,]$predict))

# When those two conditional probabilities are equal, forecast parity is achieved



install.packages(c("yardstick", "tidyverse"))
install.packages("rlang")
install.packages("vctrs")
install.packages("tibble")
install.packages("randomForest")

library(readxl)
library(ggplot2)
library(dplyr)
library(reshape2)
library(fastDummies)
library(caret)

options(scipen = 999)


pcd <- read_excel('Patient_Claim_Data.xlsx')
pcd

median_income <- read_excel('Median_Income.xlsx')


# Creating a new dataset with the required columns
transformed_data <- pcd %>%
  mutate(
    TOTAL_CHARGE = CLM_TOT_CHRG_AMT_inp + CLM_TOT_CHRG_AMT_out
  ) %>%
  select(BENE_ID, PRNCPAL_DGNS_CD_inp,PRNCPAL_DGNS_CD_out, CLM_E_POA_IND_SW1, Number_of_Claims_inp, Number_of_Claims_out, TOTAL_CHARGE, PRVDR_STATE_CD_inp) %>%
  rename(PRVDR_STATE_CD = PRVDR_STATE_CD_inp)

# Displaying the first few rows of the new dataset
head(transformed_data)
transformed_data
transformed_data <- distinct(transformed_data)

View(transformed_data)

# merging the datasets
final <- merge(transformed_data, median_income, by = "PRVDR_STATE_CD", all.x = TRUE)



final <- final %>%
  select(-PRVDR_STATE_CD)
# Displaying the first few rows of the merged data
head(final)

final$Median_Income <- as.numeric(final$Median_Income)

#View the final dataset
View(final)

# Statistical summary of dataset.
summary(final)


#number of null values in each column
colSums(is.na(final))

# Distribution of categorical variables
categorical_vars <- c("PRNCPAL_DGNS_CD_inp", "PRNCPAL_DGNS_CD_out")
for (var in categorical_vars) {
  cat(paste("Distribution of", var, ":\n"))
  print(table(final[[var]]))
}

#Frequency encoding

# Frequency encoding for PRNCPAL_DGNS_CD_inp
freq_encode_inp <- table(final$PRNCPAL_DGNS_CD_inp)
final$PRNCPAL_DGNS_CD_inp_encoded <- freq_encode_inp[as.character(final$PRNCPAL_DGNS_CD_inp)]

# Frequency encoding for PRNCPAL_DGNS_CD_out
freq_encode_out <- table(final$PRNCPAL_DGNS_CD_out)
final$PRNCPAL_DGNS_CD_out_encoded <- freq_encode_out[as.character(final$PRNCPAL_DGNS_CD_out)]

# Frequency encoding for PRNCPAL_DGNS_CD_inp
freq_encode_inp <- table(final$PRNCPAL_DGNS_CD_inp)

# Create a list of PRNCPAL_DGNS_CD codes for each unique frequency
code_list_inp <- split(names(freq_encode_inp), freq_encode_inp)

# Frequency encoding for PRNCPAL_DGNS_CD_out
freq_encode_out <- table(final$PRNCPAL_DGNS_CD_out)

# Create a list of PRNCPAL_DGNS_CD codes for each unique frequency
code_list_out <- split(names(freq_encode_out), freq_encode_out)

# Print frequency : codes format for PRNCPAL_DGNS_CD_inp
cat("PRNCPAL_DGNS_CD_inp:\n")
for (freq in names(code_list_inp)) {
  cat(freq, ":", paste(code_list_inp[[freq]], collapse = ", "), "\n")
}

# Print frequency : codes format for PRNCPAL_DGNS_CD_out
cat("\nPRNCPAL_DGNS_CD_out:\n")
for (freq in names(code_list_out)) {
  cat(freq, ":", paste(code_list_out[[freq]], collapse = ", "), "\n")
}



summary(final)
# removing null values
final <- na.omit(final)

#Creating dummy variables
final_dummy <- dummy_cols(final, select_columns = "CLM_E_POA_IND_SW1")

View(final_dummy)

# Distribution
hist(log(final_dummy$TOTAL_CHARGE))

# Linear regression model
linfinal <- final_dummy[,c('TOTAL_CHARGE', 'Median_Income', 'CLM_E_POA_IND_SW1_Y', 'Number_of_Claims_inp', 'Number_of_Claims_out', 'PRNCPAL_DGNS_CD_inp_encoded.N', 'PRNCPAL_DGNS_CD_out_encoded.N')]

hist(linfinal$TOTAL_CHARGE, xlab = 'Total Charge', main = 'Distribution of Total Charge', col = 'red')

hist(log(linfinal$TOTAL_CHARGE), xlab = 'Log values of Total Charge', main = 'Distribution of Log of Total Charge', col = 'green')

linfinal$log_TOTAL_CHARGE <- log(linfinal$TOTAL_CHARGE)

linfinal <- linfinal[, names(linfinal) != "TOTAL_CHARGE"]

summary(linfinal)
set.seed(1123)
sam_lin <- createDataPartition(linfinal$log_TOTAL_CHARGE, p = 0.7, list = FALSE)

lintrain = linfinal[sam_lin,]
lintest = linfinal[-sam_lin,]

#Linear Regression Model

lm_model <-lm(log_TOTAL_CHARGE ~ ., data = lintrain)
summary(lm_model)

# Predicted values
predicted_values_lin <- predict(lm_model, newdata = lintest)


# Calculate the residuals
residuals_lin <- lintest$log_TOTAL_CHARGE - predicted_values_lin

# Calculate the Mean Squared Error (MSE)
mse_lin <- mean(residuals_lin^2)

# Calculate the Root Mean Squared Error (RMSE)
rmse_lin <- sqrt(mse_lin)

# Print the RMSE
cat("Root Mean Squared Error (RMSE) of Linear model:", rmse_lin, "\n")


#Creating the classes

breaks <- c(-Inf, 1000, 10000, 100000, 1000000, Inf)
labels <- c(1, 2,3, 4, 5)

# Apply cut function to create classes
final_dummy$TC_class <- cut(final_dummy$TOTAL_CHARGE, breaks = breaks, labels = labels, include.lowest = TRUE)

summary(final_dummy)


final_dummy <- final_dummy[,c('TC_class', 'Median_Income', 'CLM_E_POA_IND_SW1_Y', 'Number_of_Claims_inp', 'Number_of_Claims_out', 'PRNCPAL_DGNS_CD_inp_encoded.N', 'PRNCPAL_DGNS_CD_out_encoded.N')]

# Classification model
set.seed(1123)
sam <- createDataPartition(final_dummy$TC_class, p = 0.7, list = FALSE)

train = final_dummy[sam,]
test = final_dummy[-sam,]


# KNN Model

# Defining the tuning grid
k_values <- c(3, 5, 7, 9, 11)  # You can extend or modify this list
param_grid <- expand.grid(k = k_values)

# Set up the control parameters for k-fold cross-validation
ctrl <- trainControl(method = "cv", number = 5)

# Train the kNN model using grid search
knn_model <- train(TC_class ~ ., data = train, 
                   method = "knn", 
                   trControl = ctrl,
                   tuneGrid = param_grid)

# Print the best model
print(knn_model)

# Make predictions on the test set
predictions_knn <- predict(knn_model, newdata = test)

# Evaluate the model
conf_matrix_knn <- confusionMatrix(predictions_knn, test$TC_class)
print(conf_matrix_knn)


levels(predictions)
levels(test$TC_class)

predicted_probabilities_knn <- as.matrix(predict(knn_model, newdata = test, type = "prob"))

# Combine true labels and predicted probabilities
roc_curve_knn <- multiclass.roc(true_labels, predicted_probabilities_knn)


# AUC (Area Under the Curve)
cat("AUC for KNN:", auc(roc_curve_knn), "\n")




#Decision Tree:
library(rpart)
library(rpart.plot)
library(pROC)


tree_model <- rpart(TC_class ~ ., data = train,control = rpart.control(minbucket = 10,cp=0))

class(tree_model)
#summary(tree_model)
plotcp(tree_model)
prp(tree_model, type = 2,nn = TRUE )


# Make predictions on the test set
predictions <- predict(tree_model, newdata = test, type = "class")

# Confusion Matrix
conf_matrix <- confusionMatrix(predictions, test$TC_class)
print(conf_matrix)

levels(predictions)
levels(test$TC_class)

# Assuming 'true_labels' are the true class labels and 'predicted_probabilities' are the predicted probabilities for each class
true_labels <- as.factor(test$TC_class)
predicted_probabilities_tree <- as.matrix(predict(tree_model, newdata = test, type = "prob"))

# Combine true labels and predicted probabilities
roc_curve_tree <- multiclass.roc(true_labels, predicted_probabilities_tree)


# AUC (Area Under the Curve)
cat("AUC for Decision Tree:", auc(roc_curve_tree), "\n")


# Random Forest Model
library(randomForest)

# Assuming 'train' is your training dataset and 'test' is your testing dataset

# Fit the Random Forest model
rf_model <- randomForest(TC_class ~ ., data = train, ntree = 100, importance = TRUE)

# Make predictions on the test set
predictions_rf <- predict(rf_model, newdata = test)

# Print the confusion matrix
conf_matrix_rf <- table(predictions_rf, test$TC_class)
print(conf_matrix_rf)

# Calculate accuracy
accuracy_rf <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
print(paste("Accuracy:", accuracy_rf))

# Print the variable importance
print(varImpPlot(rf_model))

# Confusion Matrix
conf_matrix_rf <- confusionMatrix(predictions_rf, test$TC_class)
print(conf_matrix_rf)

levels(predictions_rf)
levels(test$TC_class)


predicted_probabilities_rf <- as.matrix(predict(tree_model, newdata = test, type = "prob"))

# Combine true labels and predicted probabilities
roc_curve_rf <- multiclass.roc(true_labels, predicted_probabilities_rf)


# AUC (Area Under the Curve)
cat("AUC for Random Forest:", auc(roc_curve_rf), "\n")



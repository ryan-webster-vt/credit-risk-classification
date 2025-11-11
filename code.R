credit_risk <- read.csv("credit_customers.csv", stringsAsFactors = T)
library(tidyverse)
library(caret)
library(randomForest)
library(knitr)
library(cluster)

any(colSums(is.na(credit_risk)) != 0)

credit_risk <- credit_risk %>%
  # Takes the personal_status variable and splits into two new cols, sex and marital status
  separate(personal_status, into = c("sex", "marital_status"), sep = " ")

credit_risk <- credit_risk %>% 
  mutate(class = ifelse(class == "good", 1, 0))

table(credit_risk$class) %>% 
  kable(col.names = c("Class", "Count"))

cluster_data <- credit_risk %>% 
  select(duration, credit_amount, age, installment_commitment,
         residence_since, existing_credits, num_dependents)
cluster_data_scale <- scale(cluster_data)

set.seed(1)
kmeans_result <- kmeans(cluster_data_scale, centers = 3)

aggregate(cluster_data, by = list(kmeans_result$cluster), FUN = mean) %>% 
  kable(
    col.names = c(
      "Group", "Duration", "Credit Amount", "Age", 
      "Installment Commit.", "Residence Since", "Existing Credits", 
      "Num Dependents")
  )

set.seed(1)
# Split the data into 75/25 
train_index <- sample(1:nrow(credit_risk), size = 0.75 * nrow(credit_risk))
training_data <- credit_risk[train_index, ]
testing_data <- credit_risk[-train_index, ]

logit_reg_full <- glm(class ~ ., data = training_data, family = "binomial")
# summary(logit_reg_full)
# Calculate Full Model's R2
cat(
  "Full Model's R^2:",
  (1 - (logit_reg_full$deviance / logit_reg_full$null.deviance)),
  "\nFull Model's AIC:",
  logit_reg_full$aic, "\n"
)

# Null model
logit_reg_null <- glm(class ~ 1, data = training_data, family = "binomial")
# Stepwise forward
logit_reg_stepwise <- step(
  logit_reg_null,
  scope = list(lower = logit_reg_null, upper = logit_reg_full),
  direction = "forward"
)

# summary(logit_reg_stepwise)
cat(
  "Reduced Model's R^2:", 
  (1 - (logit_reg_stepwise$deviance / logit_reg_stepwise$null.deviance)),
  "\nReduced Model's AIC:", 
  logit_reg_stepwise$aic, 
  "\n"
)

# Make predictions on testing data, classify by 0.5, create confusion matrix
logit_pred <- predict(
  logit_reg_stepwise, 
  newdata = testing_data, 
  type = "response"
)
logit_pred_class <- ifelse(logit_pred > 0.5, 1, 0)
confusionMatrix(
  factor(logit_pred_class), 
  factor(testing_data$class), 
  positive = "1"
)

maximize_youden <- function(predictions) {
  thresholds <- seq(0, 1, by = 0.01)
  
  # Initialize data frame to store results
  results <- data.frame(
    threshold = thresholds,
    sensitivity = rep(0, length(thresholds)),
    specificity = rep(0, length(thresholds)),
    youden_index = rep(0, length(thresholds))
  )
  
  # Actual class labels
  actual <- factor(testing_data$class)
  
  for (i in seq_along(thresholds)) {
    # Find threshold, predict by threshold and extract sensitivity and specificity
    # data from the resulting confusion matrix. Calculate youden index.
    t <- thresholds[i]
    predicted_class <- factor(ifelse(predictions > t, 1, 0))
    cm <- confusionMatrix(predicted_class, actual, positive = "1")
    results$sensitivity[i] <- cm$byClass["Sensitivity"]
    results$specificity[i] <- cm$byClass["Specificity"]
    results$youden_index[i] <- results$sensitivity[i] + results$specificity[i] - 1
  }
  
  # Extract best threshold based on max youden index
  best_row <- results[which.max(results$youden_index), ]
  return(best_row)
}

best_row <- maximize_youden(logit_pred)
best_row

best_thresh <- best_row$threshold
logit_pred_class <- ifelse(logit_pred > best_thresh, 1, 0)
confusionMatrix(
  factor(logit_pred_class), 
  factor(testing_data$class), 
  positive = "1"
)

set.seed(1)
random_forest <- randomForest(
  as.factor(class) ~ ., 
  data = training_data, 
  mtry = dim(training_data)[2] - 1,
  importance = TRUE,
  ntree = 500
)

y_hat <- predict(random_forest, newdata = testing_data, type = "prob")
random_forest_class <- ifelse(y_hat[, 2] > 0.5, 1, 0)
confusionMatrix(
  as.factor(random_forest_class), 
  as.factor(testing_data$class),
  positive = "1"
)

best_row <- maximize_youden(y_hat[, 2])
best_row

best_thresh <- best_row$threshold

random_forest_class <- ifelse(y_hat[, 2] > best_thresh, 1, 0)
confusionMatrix(
  as.factor(random_forest_class), 
  as.factor(testing_data$class), 
  positive = "1"
)

varImpPlot(random_forest, main = "Variable Importance")

mean((as.numeric(random_forest_class) - testing_data$class)^2)

# Store variables to scale
scale_vars <- c("duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents")

# Scale
credit_risk_scaled <- credit_risk %>% 
  mutate(across(all_of(scale_vars), scale))

training_data <- credit_risk_scaled[train_index, ]
testing_data <- credit_risk_scaled[-train_index, ]

set.seed(1)
fitControl <- trainControl(method = "cv",number = 25)
knn <- train(
  as.factor(class) ~ credit_amount + checking_status + age + duration + purpose + employment,
  method = "knn",
  tuneGrid = expand.grid(k = 1:25),
  trControl = fitControl,
  metric = "Accuracy",
  data = training_data
)

plot(knn)

cat("Best K:", knn$bestTune[1, 1])

knn_pred_prob <- predict(knn, newdata = testing_data, type = "prob")
knn_pred_class <- ifelse(knn_pred_prob[, 2] > 0.5, 1, 0)
confusionMatrix(
  as.factor(knn_pred_class), 
  as.factor(testing_data$class), 
  positive = "1"
)

# Extract best threshold based on max youden index
best_row <- maximize_youden(knn_pred_prob[, 2])
best_row

best_thresh <- best_row$threshold
knn_pred_prob <- predict(knn, newdata = testing_data, type = "prob")
knn_pred_class <- ifelse(knn_pred_prob[, 2] > best_thresh, 1, 0)
confusionMatrix(
  as.factor(knn_pred_class), 
  as.factor(testing_data$class), 
  positive = "1"
)

# MSE
mean((as.numeric(knn_pred_class) - testing_data$class)^2)










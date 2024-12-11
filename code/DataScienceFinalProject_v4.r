# Data Science Final Project

# Names: Catalina Becu, Jorn Diesveld, Rebecca Hess, and Jorge Paredes

# Date: November - 2024

# [0] Preamble ----
options(repr.plot.width=15, repr.plot.height=12)

# Package installation and later call

if (!require('tidyverse')) suppressMessages(install.packages("tidyverse"))
if (!require('forcats')) suppressMessages(install.packages("forcats"))
if (!require('pheatmap')) suppressMessages(install.packages("pheatmap"))
if (!require('class')) suppressMessages(install.packages("class"))
if (!require('caret')) suppressMessages(install.packages("caret"))
if (!require('randomForest')) suppressMessages(install.packages("randomForest"))
if (!require('xgboost')) suppressMessages(install.packages("xgboost"))
if (!require('glmnet')) suppressMessages(install.packages("glmnet"))
if (!require('pROC')) suppressMessages(install.packages('pROC'))
if (!require('stargazer')) suppressMessages(install.packages('stargazer'))
if (!require('gridExtra')) suppressMessages(install.packages('gridExtra'))
if (!require('fastDummies')) suppressMessages(install.packages('fastDummies'))
if (!require('janitor')) suppressMessages(install.packages('janitor'))

suppressMessages(library(tidyverse))
suppressMessages(library(forcats))
suppressMessages(library(pheatmap))
suppressMessages(library(class))
suppressMessages(library(caret))
suppressMessages(library(randomForest))
suppressMessages(library(xgboost))
suppressMessages(library(glmnet))
suppressMessages(library(pROC))
suppressMessages(library(stargazer))
suppressMessages(library(gridExtra))
suppressMessages(library(fastDummies))
suppressMessages(library(janitor))
# [1] Data ----
## Reading, selecting variables, and making numerical some categorical variables.
set.seed(1)
data <- read.csv('https://raw.githubusercontent.com/jparedes-m/DataScienceBSE/refs/heads/main/data/credit.csv') %>%
    select(age, personal_status, job, housing, savings_status, checking_status, credit_amount, duration, purpose, credit_history, property_magnitude, housing, existing_credits, num_dependents, foreign_worker, installment_commitment, residence_since, class) %>%
    separate(personal_status, into = c("sex", "p_status"), sep = " ") %>%
    mutate(class = ifelse(class == "good", 0, 1)) %>%
    mutate(savings_status = case_when(
        savings_status == "no known savings" ~ 'no known savings',
        savings_status == "<100" ~ "little",
        savings_status == "100<=X<500" ~ 'moderate',
        savings_status == "500<=X<1000" ~ 'quite rich',
        savings_status == ">=1000" ~ 'rich',
        TRUE ~ NA)) %>%
    mutate(checking_status = case_when(
        checking_status == 'no checking' ~ "no checking",
        checking_status == "<0" ~ 'little',
        checking_status == "0<=X<200" ~ 'moderate',
        checking_status == ">=200" ~ 'rich',
        TRUE ~ NA)) %>%
  mutate(repayment_burden = credit_amount / duration) %>%
  rename(savings_account = savings_status, checking_account = checking_status) %>%
  mutate(checking_account = as.factor(checking_account), savings_account = as.factor(savings_account), property_magnitude = as.factor(property_magnitude)) %>%
  relocate(class)

# [2] Missing values / Factors treatment ----
## Is there any missing data?
sapply(data, \(x) 100*mean(is.na(x)))

# [3] Exploratory data analysis ----
## [3.1] Summary statistics ----
# Ensure 'class' is a factor
data$class <- as.factor(data$class)

cat_vars <- c("sex", "p_status", "job", "housing", "savings_account",
              "checking_account", "purpose", "credit_history",
              "property_magnitude", "foreign_worker")

melted_data <- data %>%
  select(class, all_of(cat_vars)) %>%
  pivot_longer(cols = all_of(cat_vars), names_to = "Variable", values_to = "Category") %>%
  group_by(Variable, Category, class) %>%
  summarise(Frequency = n(), .groups = "drop")

plot_list <- list()
for (var in unique(melted_data$Variable)) {
  plot_data <- melted_data %>% filter(Variable == var)

  p <- ggplot(plot_data, aes(x = Category, y = Frequency, fill = class)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = paste("Frequency of", var, "by Class"),
         x = var, y = "Frequency") +
    scale_fill_manual(values = c("salmon", "mediumaquamarine"),
                      labels = c("Bad", "Good")) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      axis.text.y = element_text(size = 10),
      axis.title.x = element_text(size = 12, face = "bold"),
      axis.title.y = element_text(size = 12, face = "bold"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      legend.position = "top",
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 10)
    )

  # Add the plot to the list
  plot_list[[var]] <- p
}

# Arrange plots into a grid layout (4 per page)
num_plots <- length(plot_list)
plots_per_page <- 4
pages <- ceiling(num_plots / plots_per_page)

for (page in 1:pages) {
  start_idx <- (page - 1) * plots_per_page + 1
  end_idx <- min(page * plots_per_page, num_plots)
  current_plots <- plot_list[start_idx:end_idx]

  capture.output({
    grid.arrange(grobs = current_plots, ncol = 2, top = paste("Page", page))
  })
}

# Calculate the proportion of each class
class_proportions <- table(data$class) / nrow(data)

# Set up a single plot
par(mfrow = c(1, 1))  # Single plot layout

# Create the bar plot
barplot(class_proportions,
        names.arg = c("Bad", "Good"),  # Class names
        main = "Class Imbalance in the Sample",  # Title
        col = c("salmon", "mediumaquamarine"),  # Bar colors
        ylab = "Proportion",  # Y-axis label
        xlab = "Class")  # X-axis label

# Add a border box around the plot
box()

# Add a grid for better visualization
grid()

rm(class_proportions, num_plots, pages, plot_list, plots_per_page, start_idx, end_idx, current_plots, melted_data, plot_data, p, page, var)

# [4] Preprocessing ----
data <- data %>% mutate(class = as.numeric(class)-1)
# Create dummy columns
df_cat <- dummy_cols(
  data,
  select_columns = cat_vars,
  remove_first_dummy = FALSE, remove_most_frequent_dummy = TRUE,  ignore_na = FALSE,
  remove_selected_columns = TRUE,
  omit_colname_prefix = FALSE) %>% 
  # clean column names
  janitor::clean_names() %>% 
  mutate_if(is.integer, as.numeric)

sapply(df_cat, \(x) class(x))

map <- pheatmap(cor(df_cat, use = "complete.obs"),
                display_numbers = TRUE,
                number_color = "black",
                main = "Feature Correlation Heatmap", treeheight_row = F, treeheight_col = F)

# Specify the variable of interest
target_variable <- "class"

# Calculate correlation with the rest of the variables
correlations <- sapply(df_cat, function(x) cor(df_cat[[target_variable]], x, use = "complete.obs"))
correlations <- sort(correlations, decreasing = TRUE)

# Analysis of the variables that likely need transformation of some kind
numerical_vars <- c("age", "credit_amount", "repayment_burden", "duration", "installment_commitment", "num_dependents", "existing_credits", "residence_since")

# Normalize the selected variables (min-max scaling)
data_normalized <- data %>%
  mutate(across(all_of(numerical_vars), ~ (. - min(.)) / (max(.) - min(.))))

# Reshape data to long format
data_normalized <- data_normalized %>%
  pivot_longer(cols = all_of(numerical_vars), names_to = "Variable", values_to = "Value")

ggplot(data_normalized, aes(x = Variable, y = Value, fill = as.factor(class))) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  labs(
    title = "Normalized Boxplots Numerical Variables by Class",
    x = "Variable",
    y = "Normalized Value",
    fill = "Class"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_x_discrete(guide = guide_axis(angle = 45)) +  # Rotate x-axis labels for clarity
  scale_fill_manual(
    values = c("0" = "steelblue", "1" = "red"),
    labels = c("0" = "Good", "1" = "Bad")
  )

plot(histogram(data$num_dependents))
plot(histogram(data$existing_credits))

# Generate smooth plots with transformations
variables <- c("age", "credit_amount", "repayment_burden", "duration", "installment_commitment", "existing_credits")
transformations <- list(identity = function(x) x, log = function(x) log(x), squared = function(x) x^2)

smooth_plots <- list()
for (var in variables) {
  for (trans_name in names(transformations)) {
    trans <- transformations[[trans_name]]
    suppressMessages(suppressWarnings({
      smooth_plots[[paste(var, trans_name, sep = "_")]] <- ggplot(data, aes(x = trans(.data[[var]]), y = as.numeric(class))) +
        geom_point(alpha = 0.5) +
        geom_smooth(method = "loess", show.legend = FALSE, se = FALSE) +
        labs(
          x = paste0(trans_name, "(", var, ")"),
          y = "Class (Numeric)",
          title = paste("Smoothed Plot of", trans_name, "(", var, ")")
        ) +
        theme_minimal()
    }))
  }
}

# Arrange smooth plots into grids (4 per page)
plots_per_page <- 4
total_plots <- length(smooth_plots)
pages <- ceiling(total_plots / plots_per_page)

for (page in 1:pages) {
  start_idx <- (page - 1) * plots_per_page + 1
  end_idx <- min(page * plots_per_page, total_plots)
  suppressMessages(suppressWarnings({
    grid.arrange(grobs = smooth_plots[start_idx:end_idx], ncol = 2, top = paste("Page", page))
  }))
}

rm(data_normalized, map, numerical_vars, correlations, end_idx, page, pages, plots_per_page, smooth_plots, start_idx, total_plots, trans, trans_name, transformations, var, variables)

# Feature Engineering
df <- data
df <- df[df$repayment_burden < 1000,] # Remove outliers in repayment_burden
df <- df %>% mutate(age = log(1 + age)) # Log-transform age
df <- df %>% mutate(duration = log(1 + duration)) # Log-transform duration
df <- df %>% mutate(
  credit_amount2 = credit_amount^2, 
  num_dependents2 = num_dependents^2, 
  repayment_burden2 = repayment_burden^2
)

# Standardize numeric variables
num_vars <- c('credit_amount', 'installment_commitment', 'residence_since', 'existing_credits', 'num_dependents', 'credit_amount2', 'num_dependents2', 'repayment_burden', 'repayment_burden2')
df <- df %>% mutate(across(all_of(num_vars), scale))

# [5] Models ----
## [5.0] Train-test split ----
train_index <- sample(1:nrow(df), 0.75 * nrow(df))
X <- df %>% select(-class)
y <- factor(df$class, levels = c(0,1), labels = c("good", "bad"))

train_x <- X[train_index, ]
train_y <- y[train_index]

test_x <- X[-train_index, ]
test_y <- y[-train_index]

# how to adress class imbalance:
train_data <- cbind(train_x, class = train_y)
majority <- train_data %>% filter(class == "good")
minority <- train_data %>% filter(class == "bad")
majority_undersampled <- majority %>% sample_n(nrow(minority))

train_data_balanced <- bind_rows(majority_undersampled, minority)
train_data_balanced <- train_data_balanced %>% sample_frac(1)

train_x <- train_data_balanced %>% select(-class)
train_y <- train_data_balanced$class

## [5.1] KNN ----
control <- trainControl(method = "cv", number = 7, classProbs = TRUE)
knn_model <- train(class ~ ., data = train_data_balanced, method = "knn", tuneGrid = data.frame(k = 1:25), trControl = control)

best_k <- knn_model$bestTune$k
# instead of hard classification we will use the class probabilities
knn_predictions <- predict(knn_model, newdata = test_x, type = "raw")
conf_matrix_knn <- confusionMatrix(knn_predictions, test_y)

knn_probabilities <- predict(knn_model, newdata = test_x, type = "prob")
# we are interested in predicting the 'bad' class
roc_knn <- roc(test_y, knn_probabilities[, "bad"])

## [5.2] Random Forest ----
ntree_grid  <- 100*c(1:5)
mtry_grid <- 1:(ncol(train_data_balanced) - 1)
results <- expand.grid(ntree = ntree_grid, mtry = mtry_grid)
results$oob_accuracy <- NA
results$test_accuracy <- NA

for (i in 1:nrow(results)) {
    ntree <- results$ntree[i]
    mtry <- results$mtry[i]
    
    rf <- randomForest(
        x = train_x, 
        y = train_y, 
        mtry = mtry, 
        ntree = ntree
    )
    
    # Predictions on validation set
    y_pred_val <- predict(rf, test_x)
    
    # Store OOB and test accuracy
    results$oob_acc[i] <- 1 - mean(rf$err.rate[, "OOB"])
    results$test_acc[i] <- mean(test_y == y_pred_val)
}

# Find the best combination based on test accuracy
best_index <- which.max(results$test_acc)
best_ntree <- results$ntree[best_index]
best_mtry <- results$mtry[best_index]
cat("Best ntree:", best_ntree, ", Best mtry:", best_mtry, "\n")

rf_model <- randomForest(
  class ~ ., 
  data = train_data_balanced, 
  mtry = best_mtry,  
  ntree = best_ntree,     
  importance = TRUE
)

rf_predictions <- predict(rf_model, newdata = test_x)
conf_matrix_rf <- confusionMatrix(rf_predictions, test_y)
rf_probabilities <- predict(rf_model, newdata = test_x, type = "prob")
roc_rf <- roc(test_y, rf_probabilities[, "bad"])

### [5.2.1] Variable importance ----
importance_rf <- importance(rf_model) %>% as.data.frame() %>%
  mutate(Normalized = MeanDecreaseGini / max(MeanDecreaseGini, na.rm = TRUE)) %>%
  arrange(desc(Normalized)) %>%
  select(Normalized) %>% 
  rownames_to_column(importance_df, "Variable")

# Step 4: Plot the variable importance
ggplot(importance_rf, aes(x = reorder(Variable, -Normalized), y = Normalized)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(
    title = "Random Forest Variable Importance",
    x = "Variable",
    y = "Normalized Importance"
  ) +
  theme_minimal()

## [5.3] XGBoost ----
train_matrix <- model.matrix(class ~ ., data = train_data_balanced)[, -1]
test_matrix <- model.matrix(~ ., data = test_x)[, -1]
train_label <- train_y
test_label <- test_y

xgb_control <- trainControl(method = "cv", number = 7, verboseIter = FALSE, classProbs = TRUE, summaryFunction = twoClassSummary)

xgb_model <- train(x = train_matrix, y = train_label, method = "xgbTree", trControl = xgb_control, verbosity = 0, metric = 'ROC',
    tuneGrid = expand.grid(nrounds = seq(50, 200, by = 50), max_depth = c(3, 6), eta = c(0.1), gamma = c(0, 0.1), colsample_bytree = 0.8, min_child_weight = c(1,3), subsample = 0.8))

xgb_predictions <- predict(xgb_model, newdata = test_matrix)
conf_matrix_xgb <- confusionMatrix(xgb_predictions, test_label)

xgb_probabilities <- predict(xgb_model, newdata = test_matrix, type = "prob")

roc_xgb <- roc(test_label, xgb_probabilities[, "bad"])
### [5.3.1] Variable importance ----
importance_xgb <- varImp(xgb_model)$importance %>% 
  as.data.frame() %>%
  rownames_to_column(var = "Feature") %>%
  mutate(NormalizedImportance = Overall / max(Overall)) %>%
  arrange(desc(NormalizedImportance)) %>%
  select(Feature, NormalizedImportance) %>% 
  filter(NormalizedImportance > 0)

ggplot(importance_xgb, aes(x = reorder(Feature, -NormalizedImportance), y = NormalizedImportance)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(title = "XGBoost Variable Importance",
       x = "Variable", y = "Normalized Importance") +
  theme_minimal()

## [5.4] Elastic net with logit ----
train_matrix <- model.matrix(class ~ ., data = train_data_balanced)[, -1]
test_matrix <- model.matrix(~ ., data = test_x)[, -1]
train_label <- as.numeric(train_y)-1
test_label <- as.numeric(test_y)-1

### [5.4.0] Cross-validation to find the best alpha ----
results <- lapply(seq(0, 1, length.out = 51)[-c(1, 51)], function(alpha) {  # Exclude alpha = 0 and alpha = 1
    model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = alpha, nfolds = 7)
    probabilities <- predict(model, newx = test_matrix, s = "lambda.min", type = "response")
    auc <- suppressMessages(roc(test_label, as.vector(probabilities))$auc)
    list(alpha = alpha, model = model, lambda = model$lambda.min, auc = auc)
})
results_df <- do.call(rbind, lapply(results, function(x) data.frame(alpha = x$alpha, lambda = x$lambda, auc = x$auc)))
best_alpha <- results_df[which.max(results_df$auc), "alpha"]
### [5.4.1] Ridge Model ----
ridge_model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = 0, lambda = NULL, nfolds = 7)
ridge_probabilities <- predict(ridge_model, newx = test_matrix, s = "lambda.min", type = "response")
ridge_predictions <- predict(ridge_model, newx = test_matrix, s = "lambda.min", type = "class")
ridge_predictions <- factor(ridge_predictions, levels = c(0, 1))
test_label_factor <- factor(test_label, levels = c(0, 1))
conf_matrix_ridge <- confusionMatrix(ridge_predictions, test_label_factor)
roc_ridge <- roc(test_label, as.numeric(ridge_probabilities))

#### [5.4.1.1] Shrinkage
ridge_coefs_1se <- coef(ridge_model, s = "lambda.1se") %>% as.matrix() %>% as.data.frame() %>% rownames_to_column("variable") %>% rename(lambda_1se = s1)
ridge_coefs_min <- coef(ridge_model, s = "lambda.min") %>% as.matrix() %>% as.data.frame() %>% rownames_to_column("variable") %>% rename(lambda_min = s1)
ridge_coefs <- left_join(ridge_coefs_1se, ridge_coefs_min, by = "variable") %>% filter(lambda_min != 0 | lambda_1se != 0)
### [5.4.2] Elastic net ----
elastic_net_model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = alpha, lambda = NULL, nfolds = 7)
elastic_net_probabilities <- predict(elastic_net_model, newx = test_matrix, s = "lambda.min", type = "response")
elastic_net_predictions <- predict(elastic_net_model, newx = test_matrix, s = "lambda.min", type = "class")
elastic_net_predictions <- factor(elastic_net_predictions, levels = c(0, 1))
test_label_factor <- factor(test_label, levels = c(0, 1))
conf_matrix_elastic <- confusionMatrix(elastic_net_predictions, test_label_factor)
roc_elastic <- roc(test_label, as.numeric(elastic_net_probabilities))

#### [5.4.2.1] Shrinkage 
elastic_coefs_1se <- coef(elastic_net_model, s = "lambda.1se")%>% as.matrix() %>% as.data.frame() %>% rownames_to_column("variable") %>% rename(lambda_1se = s1)
elastic_coefs_min <- coef(elastic_net_model, s = "lambda.min")%>% as.matrix() %>% as.data.frame() %>% rownames_to_column("variable") %>% rename(lambda_min = s1)
elastic_coefs <- left_join(elastic_coefs_1se, elastic_coefs_min, by = "variable") %>% filter(lambda_min != 0 | lambda_1se != 0)

### [5.4.3] Lasso ----
lasso_model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = 1, lambda = NULL, nfolds = 7)
lasso_probabilities <- predict(lasso_model, newx = test_matrix, s = "lambda.min", type = "response")
lasso_predictions <- predict(lasso_model, newx = test_matrix, s = "lambda.min", type = "class")
lasso_predictions <- factor(lasso_predictions, levels = c(0, 1))
test_label_factor <- factor(test_label, levels = c(0, 1))
conf_matrix_lasso <- confusionMatrix(lasso_predictions, test_label_factor)
roc_lasso <- roc(test_label, as.numeric(lasso_probabilities))

#### [5.4.3.1] Shrinkage
lasso_coefs_min <- coef(lasso_model, s = "lambda.min")%>% as.matrix() %>% as.data.frame() %>% rownames_to_column("variable") %>% rename(lambda_min = s1)
lasso_coefs_1se <- coef(lasso_model, s = "lambda.1se")%>% as.matrix() %>% as.data.frame() %>% rownames_to_column("variable") %>% rename(lambda_1se = s1)
lasso_coefs <- left_join(lasso_coefs_1se, lasso_coefs_min, by = "variable") %>% filter(lambda_min != 0 | lambda_1se != 0)
### [5.4.4] Logit without penalization ----
control <- trainControl(method = "cv", number = 7, classProbs = TRUE, summaryFunction = twoClassSummary)
logit_model <- train(class ~ ., data = train_data_balanced, family=binomial(logit), method = 'glm',trControl = control)
logit_probabilities <- predict(logit_model, newdata = test_x, type = "prob")
logit_predictions <- predict(logit_model, newdata = test_x, type = 'raw')
test_label <- factor(test_label, label = c('good', 'bad'), levels = c(0, 1))
conf_matrix_logit <- confusionMatrix(logit_predictions, test_label)
roc_logit <- roc(as.numeric(test_label)-1, logit_probabilities[,'bad'])

# [6] Results ----
## [6.1] Accuracy ----
# Get all the confusion matrices in a list for comparison
confusion_matrices <- list(knn = conf_matrix_knn, rf = conf_matrix_rf, xgb = conf_matrix_xgb, ridge = conf_matrix_ridge, elastic_net = conf_matrix_elastic, lasso = conf_matrix_lasso, logit = conf_matrix_logit)

## Get the accuracy, and the CI into a dataframe
accuracy <- data.frame(model = names(confusion_matrices),
                      accuracy = sapply(confusion_matrices, function(x) x$overall["Accuracy"]),
                      lower_ci = sapply(confusion_matrices, function(x) x$overall["AccuracyLower"]),
                      upper_ci = sapply(confusion_matrices, function(x) x$overall["AccuracyUpper"]))
rownames(accuracy) <- NULL
accuracy  <- accuracy %>% mutate(model = case_when(model == "knn" ~ "K-Nearest-Neighbors",
                                                   model == "rf" ~ "Random Forest",
                                                   model == "xgb" ~ "XGBoost",
                                                   model == "ridge" ~ "Ridge",
                                                   model == "elastic_net" ~ "Elastic Net",
                                                   model == "lasso" ~ "Lasso",
                                                   model == 'logit' ~ "Logit")) %>%
            mutate(model = fct_relevel(model, "Logit", "Lasso", "Elastic Net", "Ridge", "XGBoost", "Random Forest", "K-Nearest-Neighbors"))

# plot this results in a geom_point plot with error bars
ggplot(accuracy, aes(x = model, y = accuracy)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.2) +
    labs(y = "Accuracy", x = "Model", title = "Model Comparison") +
    theme_light() + coord_flip() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
          strip.text.x = element_text(face = "bold", color = "black", size = 12),
          axis.title.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(angle = 270, vjust = 0.5, hjust=1)) +
    scale_y_continuous(n.breaks = 20)

## [6.2] ROC Curves ----
# Plot all the roc curves with the AUC in the legend with the plot function
plot(y=roc_knn$sensitivities, x=1-roc_knn$specificities, col = "red", lwd = 2, type = "l", main = "ROC Curves", xlab = "False Positive Rate", ylab = "True Positive Rate")
lines(y=roc_rf$sensitivities, x=1-roc_rf$specificities, col = "blue", lwd = 2)
lines(y=roc_xgb$sensitivities, x=1-roc_xgb$specificities, col = "green", lwd = 2)
lines(y=roc_ridge$sensitivities, x=1-roc_ridge$specificities, col = "purple", lwd = 2)
lines(y=roc_elastic$sensitivities, x=1-roc_elastic$specificities, col = "orange", lwd = 2)
lines(y=roc_lasso$sensitivities, x=1-roc_lasso$specificities, col = "black", lwd = 2)
lines(y=roc_logit$sensitivities, x=1-roc_logit$specificities, col = "brown", lwd = 2)
legend("bottomright",
       legend = c(
         paste0("K-Nearest-Neighbors (AUC = ", round(auc(roc_knn), 3), ")"),
         paste0("Random Forest (AUC = ", round(auc(roc_rf), 3), ")"),
         paste0("XGBoost (AUC = ", round(auc(roc_xgb), 3), ")"),
         paste0("Ridge (AUC = ", round(auc(roc_ridge), 3), ")"),
         paste0("Elastic Net (AUC = ", round(auc(roc_elastic), 3), ")"),
         paste0("Lasso (AUC = ", round(auc(roc_lasso), 3), ")"),
          paste0("Logit (AUC = ", round(auc(roc_logit), 3), ")")
       ),
       col = c("red", "blue", "green", "purple", "orange", "black", "brown"),
       lwd = 2)

## [6.3] Shrinkage plots and tables ----
par(mfrow = c(3, 1))
plot(ridge_model)
title("Ridge Coefficient Shrinkage Plot\n")
plot(elastic_net_model)
title("Elastic Net Coefficient Shrinkage Plot\n")
plot(lasso_model)
title("Lasso Coefficient Shrinkage Plot\n")
par(mfrow = c(1, 1))

### [6.3.1] Shrinkage Coefficients
ridge_coefs
elastic_coefs
lasso_coefs
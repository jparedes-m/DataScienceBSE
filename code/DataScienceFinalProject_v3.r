# Data Science Final Project

# Names: Catalina Becu, Jorn Diesveld, Rebecca Hess, and Jorge Paredes

# Date: November - 2024

# [0] Preamble ----
set.seed(1)
library(tidyverse)
library(forcats)
library(pheatmap)
library(class)
library(caret)
library(randomForest)
library(xgboost)
library(glmnet)
library(pROC)
library(stargazer)
library(gridExtra)
library(fastDummies)
library(janitor)

# [1] Data ----
## Reading, selecting variables, and making numerical some categorical variables.

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

df <- data

## Missing data treatment 
sapply(df, \(x) 100*mean(is.na(x)))

# [3] Exploratory data analysis ----
## [3.1] Summary statistics [CATA] ----
# Ensure 'class' is a factor
data$class <- as.factor(data$class)

# Updated list of categorical variables
cat_vars <- c("sex", "p_status", "job", "housing", "savings_account", 
              "checking_account", "purpose", "credit_history", 
              "property_magnitude", "foreign_worker")

# Melt the dataset for easier plotting
melted_data <- data %>%
  select(class, all_of(cat_vars)) %>%
  pivot_longer(cols = all_of(cat_vars), names_to = "Variable", values_to = "Category") %>%
  group_by(Variable, Category, class) %>%
  summarise(Frequency = n(), .groups = "drop")

# Create a list to store the plots
plot_list <- list()

# Generate a barplot for each variable
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
  # Select plots for the current page
  start_idx <- (page - 1) * plots_per_page + 1
  end_idx <- min(page * plots_per_page, num_plots)
  current_plots <- plot_list[start_idx:end_idx]
  
  # Arrange the selected plots in a grid
  grid_plot <- grid.arrange(grobs = current_plots, ncol = 2, top = paste("Page", page))
  
  # Print the grid plot
  print(grid_plot)
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

# [4] Preprocessing ----
categorical_vars <- c("sex","p_status", "housing", "job", "savings_account", "checking_account", "purpose", "credit_history", "property_magnitude", "foreign_worker")
# Create dummy columns
df <- dummy_cols(
  df,
  select_columns = categorical_vars,
  remove_first_dummy = FALSE,          # Keeps all dummy variables
  remove_most_frequent_dummy = FALSE, # Retain all dummy levels
  ignore_na = FALSE,                  # Includes NAs if present
  remove_selected_columns = TRUE,    # Keeps original columns
  omit_colname_prefix = FALSE         # Keeps the column name as prefix
)

# Clean column names
df <- df %>%
  janitor::clean_names()

map <- pheatmap(cor(df, use = "complete.obs"), 
         display_numbers = TRUE, 
         number_color = "black",
         main = "Feature Correlation Heatmap", treeheight_row = F, treeheight_col = F)

save_pheatmap <- function(x, filename, width=12, height=12){
  stopifnot(!missing(x))
  stopifnot(!missing(filename))
  if(grepl(".png",filename)){
    png(filename, width=width, height=height, units = "in", res=300)
    grid::grid.newpage()
    grid::grid.draw(x$gtable)
    
  }
  else if(grepl(".pdf",filename)){
    pdf(filename, width=width, height=height)
    grid::grid.newpage()
    grid::grid.draw(x$gtable)
    
  }
  else{
    print("Filename did not contain '.png' or '.pdf'")
  }
}
#save_pheatmap(map, "/Users/rebeccahess/Documents/BSE Sem 1/datascience/heatmap.png", width = 10, height = 10)
# Specify the variable of interest
target_variable <- "class"
# Calculate correlation with the rest of the variables
correlations <- sapply(df, function(x) cor(df[[target_variable]], x, use = "complete.obs"))
sorted_correlations <- sort(correlations, decreasing = TRUE)
print(sorted_correlations)

#analysis of the variables that likely need transformation of some kind
# List of variables to include in the boxplots
numerical_vars <- c("age", "credit_amount", "repayment_burden", "duration", "installment_commitment", "num_dependents", "existing_credits", "residence_since")

# Normalize the selected variables (min-max scaling)
data_normalized <- data %>%
  mutate(across(all_of(numerical_vars), ~ (. - min(.)) / (max(.) - min(.))))

# Reshape data to long format
data_long <- data_normalized %>%
  pivot_longer(cols = all_of(numerical_vars), names_to = "Variable", values_to = "Value")

ggplot(data_long, aes(x = Variable, y = Value, fill = as.factor(class))) +
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

#log wins age!!!
ggplot(data, aes(x = age, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Age", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = log(age), y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Log Age", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = age^2, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Age Squared", y = "Class (Numeric)") +
  theme_minimal()

## Squared wins credit amount 
ggplot(data, aes(x = credit_amount, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Credit Amount", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = log(credit_amount), y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Log Credit Amount", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = credit_amount^2, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Credit Amount Squared", y = "Class (Numeric)") +
  theme_minimal()

#repayment burden should be squared!!!!
ggplot(data, aes(x = repayment_burden, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Credit Amount", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = log(repayment_burden), y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Log Credit Amount", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = repayment_burden^2, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Credit Amount Squared", y = "Class (Numeric)") +
  theme_minimal()

#log wins duration!!!!
ggplot(data, aes(x = duration, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Duration", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = log(duration), y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Log Duration", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = duration^2, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Duration Squared", y = "Class (Numeric)") +
  theme_minimal()

#installment_commitment!!!! all linear who cares 
ggplot(data, aes(x = installment_commitment, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Installment Com", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = log(installment_commitment), y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Log Installment Com", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = installment_commitment^2, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Installment Com Squared", y = "Class (Numeric)") +
  theme_minimal()

#installment_commitment!!!! all linear who cares 
ggplot(data, aes(x = existing_credits, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "existing_credits ", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = log(existing_credits), y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "Log existing_credits ", y = "Class (Numeric)") +
  theme_minimal()

ggplot(data, aes(x = existing_credits^2, y = as.numeric(class))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(x = "existing_credits  Squared", y = "Class (Numeric)") +
  theme_minimal()

#repayment burden has 3 people about 1,000 which is unrepresentative/extremely above the Q3 of the sample (outliers)
df <- df[df$repayment_burden<1000,]
#taking the log of age due to is right skew 
df <- df %>% mutate(age=log(1+age))
#taking the log of duration 
df <- df %>% mutate(duration = log(1+duration))

## Feature engeenering
### Add: squared age, squared credit amount, squared duration, and squared number of dependents
#df <- df %>% mutate(age2 = age^2, credit_amount2 = credit_amount^2, duration2 = duration^2, num_dependents2 = num_dependents^2)
df <- df %>% mutate(credit_amount2 = credit_amount^2, num_dependents2 = num_dependents^2, repayment_burden2 = repayment_burden^2)

## Standardization across the numeric variables
#num_vars <- c('duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 'num_dependents', 'age2', 'credit_amount2', 'duration2', 'num_dependents2', 'repayment_burden')
#removed age and age^2 -- don't need to normalize log 
#source: chatgpt said scaling doesn't matter in tree models
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
table(train_data_balanced$class)

train_x <- train_data_balanced %>% select(-class)
train_y <- train_data_balanced$class

## [5.1] KNN ----
control <- trainControl(method = "cv", number = 7, classProbs = TRUE)
knn_model <- train(class ~ ., data = train_data_balanced, method = "knn", tuneGrid = data.frame(k = 1:25), trControl = control, metric = 'ROC')

best_k <- knn_model$bestTune$k
# instead of hard classification we will use the class probabilities
knn_predictions <- predict(knn_model, newdata = test_x, type = "raw")
conf_matrix_knn <- confusionMatrix(knn_predictions, test_y)

knn_probabilities <- predict(knn_model, newdata = test_x, type = "prob")
# we are interested in predicting the 'bad' class
roc_knn <- roc(test_y, knn_probabilities[, "bad"])

## [5.2] Random Forest ----
ntree_grid  <- 100*c(1:5)
mtry_grid <- c(2, 3, floor(sqrt(ncol(train_data_balanced) -1)), 5)
rf_results <- list()
control <- trainControl(method = "cv", number = 7, classProbs = TRUE, summaryFunction = twoClassSummary)

for(ntree in ntree_grid){
    rf_grid <- expand.grid(mtry = mtry_grid)
    rf_tuned <- train(
        class ~ ., 
        data = train_data_balanced, 
        method = "rf",
        trControl = control,
        tuneGrid = rf_grid, 
        metric = 'ROC',
        ntree = ntree 
    )
    
    rf_results[[as.character(ntree)]] <- list(
        ntree = ntree,
        model = rf_tuned,
        best_mtry = rf_tuned$bestTune$mtry,
        accuracy = max(rf_tuned$results$Accuracy),
        results = rf_tuned$results
    )
}

# Find the best ntree and mtry combination based on accuracy
best_ntree <- names(which.max(sapply(rf_results, function(x) x$accuracy)))
best_result <- rf_results[[best_ntree]]

rf_model <- randomForest(class ~ .,  data = train_data_balanced, ntree = best_result$ntree, mtry = best_result$best_mtry)
rf_predictions <- predict(rf_model, newdata = test_x)
conf_matrix_rf <- confusionMatrix(rf_predictions, test_y)
rf_probabilities <- predict(rf_model, newdata = test_x, type = "prob")
roc_rf <- roc(test_y, rf_probabilities[, "bad"])

### [5.2.1] Variable importance ----
importance_table <- importance(rf_model)
importance_df <- as.data.frame(importance_table)

Normalized <- importance_df %>%
  mutate(Normalized = MeanDecreaseGini / max(MeanDecreaseGini, na.rm = TRUE)) %>%
  arrange(desc(Normalized)) %>%
  select(Normalized)
# Keep only the desired columns

# Add variable names as a column
Normalized <- tibble::rownames_to_column(Normalized, "Variable")

# View the formatted tablex
print(Normalized)
# Generate LaTeX table using stargazer
stargazer(as.matrix(Normalized), type = "latex", summary = FALSE,
          title = "Random Forest Variable Importance", align = TRUE)

# Step 4: Plot the variable importance
ggplot(Normalized, aes(x = reorder(Variable, -Normalized), y = Normalized)) +
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
importance <- varImp(xgb_model)
importance_df <- as.data.frame(importance$importance)
importance_df$Feature <- rownames(importance_df)
importance_df <- importance_df %>%
  mutate(NormalizedImportance = Overall / max(Overall)) %>%
  arrange(desc(NormalizedImportance)) %>%
  select(Feature, NormalizedImportance)

stargazer(as.matrix(Normalized), type = "latex", summary = FALSE,
          title = "XGBoost Variable Importance", align = TRUE)

ggplot(Normalized, aes(x = reorder(Variable, -Normalized), y = Normalized)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(
    title = "XGBoost Variable Importance",
    x = "Variable",
    y = "Normalized Importance"
  ) +
  theme_minimal()

## [5.4] Elastic net with logit ----
train_matrix <- model.matrix(class ~ ., data = train_data_balanced)[, -1]
test_matrix <- model.matrix(~ ., data = test_x)[, -1]
train_label <- as.numeric(train_y)-1
test_label <- as.numeric(test_y)-1

### [5.4.0] Cross-validation to find the best alpha ----
models <- list()
results <- data.frame(alpha = numeric(), lambda = numeric(), auc = numeric())
for (i in 0:50) {
    name <- paste0("alpha", i / 50)
    model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = i / 50, lambda = NULL, nfolds = 7)
    models[[name]] <- model
    probabilities <- predict(model, newx = test_matrix, s = "lambda.min", type = "response")
    auc <- suppressMessages(auc(test_label, probabilities))
    results <- rbind(results, data.frame(alpha = i / 50, lambda = model$lambda.min, auc = auc))
}
results <- results %>% filter(alpha != 0 & alpha != 1) %>% arrange(desc(auc))
alpha <- results[1, "alpha"]
### [5.4.1] Ridge Model ----
ridge_model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = 0, lambda = NULL, nfolds = 7)
ridge_probabilities <- predict(ridge_model, newx = test_matrix, s = "lambda.min", type = "response")
ridge_predictions <- predict(ridge_model, newx = test_matrix, s = "lambda.min", type = "class")
ridge_predictions <- factor(ridge_predictions, levels = c(0, 1))
test_label_factor <- factor(test_label, levels = c(0, 1))
conf_matrix_ridge <- confusionMatrix(ridge_predictions, test_label_factor)
roc_ridge <- roc(test_label, ridge_probabilities)

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
roc_elastic <- roc(test_label, as.vector(elastic_net_probabilities))

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
roc_lasso <- roc(test_label, as.vector(lasso_probabilities))

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
# End-of-File ----
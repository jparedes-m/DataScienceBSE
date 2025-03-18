# Foundations of Data Science @ BSE
Foundations of data science final project: Term 1 2024-2025 at Barcelona School of Economics

Names: Jorge Paredes, Catalina Becu, Rebecca Hess, Jorn Diesveld

## Aim of the project:

Predict using diverse methods a factor variable showing if a client is good or bad using the German credit dataset found at: https://www.openml.org/search?type=data&sort=runs&status=active&id=31

## Abstract
This project analyzes credit risk prediction using the German Credit Dataset, classifying individuals as good or bad credit risks based on financial and demographic factors. Seven machine learning models were evaluated: K-Nearest Neighbors (KNN), Random Forest, XGBoost, Logistic Regression, and its variations with Ridge, Lasso, and Elastic Net regularization. The models were compared using accuracy and ROC-AUC metrics. Random Forest performs best, with the highest accuracy and AUC (0.813), making it the most reliable model for distinguishing between “good” and “bad” credit risks. XGBoost and Elastic Net are competitive alternatives, while K-Nearest Neighbors shows the weakest performance (AUC 0.716). These findings highlight the importance of model evaluation in credit risk prediction and demonstrate that nonlinear approaches consistently outperform linear models in Supervised Learning for prediction. This analysis provides valuable insights into how machine learning can improve decision-making in financial contexts.

## Introduction
Credit risk assessment is a fundamental process for financial institutions, as it determines whether an individual is likely to meet their financial obligations. Accurate predictions in this area not only reduce potential losses but also help allocate credit resources more effectively. In this project, we use the German Credit Dataset, which consists of 1,000 observations. Each row represents an individual described by financial and demographic attributes, and a column classifying them as either a good or bad credit risk. This dataset provides a practical framework for evaluating the predictive power of machine learning models in a real-world financial setting.

The primary objective of this project is to predict an individual’s creditworthiness based on their attributes by employing and comparing different classification models. These models enable us to analyze patterns in the data, offering insights into how machine learning can be used to enhance decision-making processes in credit scoring systems. By comparing multiple models, we aim to determine which is most effective in this context, balancing accuracy, interpretability, and computational efficiency.

The machine learning models we test include both classical and modern approaches, such as distance-based algorithms (K-Nearest Neighbors), tree-based ensemble methods (Random Forest and XGBoost), and logistic regression enhanced with regularization techniques (Ridge, Lasso, and Elastic Net). Each model was chosen for its potential to address different aspects of the dataset. For example, ensemble methods are particularly suited for capturing complex feature interactions, while regularized logistic regression provides a transparent, interpretable framework for making predictions. This combination of techniques allows us to explore the balance between accuracy, complexity, and practical utility in credit scoring applications.

To evaluate the performance of these models, we rely on accuracy rates as well as the receiver operating characteristic. These metrics allow us to understand the trade-offs between sensitivity and specificity for each model. We calculate and compare the rates to assess their predictive capabilities. To enhance the interpretability of our results, we also include visualizations, such as accuracy plots, which clearly display the comparative performance of the models.

In conclusion, this project demonstrates how machine learning can be leveraged to improve credit risk classification. Our analysis emphasizes the importance of selecting appropriate models based on the specific requirements of a given application, such as balancing the need for high accuracy with the demand for transparency and ease of implementation. Ultimately, our findings highlight the potential of data science in addressing critical challenges in financial services, which contribute to more efficient and equitable credit allocation.

## Data Description
The dataset provided is a German credit dataset containing 1,000 entries, each representing an individual’s credit information. It includes 19 variables related to personal, financial, and credit-related aspects. The data set captures individual attributes such as age, sex, marital status, housing type, residency duration, number of dependents, and foreign worker status. Key financial variables include balances in both savings and checking accounts, credit history, and the number of existing credits held at the same bank. We have credit request details including the requested credit amount, the duration of the credit, the purpose of the credit request, the property magnitude (e.g., life insurance, real estate, or car) that can be leveraged against the credit request, and installment commitment, representing the individual’s monthly disposable income allocated towards credit payments. Additionally, we have the repayment burden calculated as the ratio of the credit amount to its duration, representing the individual’s temporal payment obligation. Finally, the variable we are trying to predict is ”class”: the classification of the credit request as "good" or "bad".

To analyze the relationships between variables, we first converted the categorical variables into binary variables, leaving out the most frequent group as the reference category.

Now we graph the frequency of the categorical variables for each class to see if there are features that may help us distinguish between classes.

```
Graphs to be found in RMD or Jupyter notebooks in code section
```
From these graphs we note that foreign workers, male, radio/TV purpose and singles have more frequency in bad class, which could be a good insight for prediction.

Then we analyze the correlation of the predictors with each other as well as with our response variable “class” in a correlation heatmap.

We can see that the variables with a correlation absolute value of ρ above or equal to 0.15 with the “class” variable are: duration and credit amount, checking account ”little” or no checking account, credit history - critical/other existing credit and savings account ”little”, which is intuitive in the sense of predicting risky clients.

By normalizing the numerical variables to a scale of 0 to 1 using their respective minimum and maximum values, we are able to plot their boxplots on the same axis. Grouping them by credit classification, we can easily see how the distributions of each variable differ depending on whether they are considered “good” or “bad.” This offers some insight into the impact of the numerical variables on the credit classification. Additionally, regardless of credit classification, we can see the distributions are right-skewed for age, credit_amount, duration, and repayment_burden, which we will address in the cleaning and preparation section. We can also easily identify outliers in the repayment burden variable.

As we can see, 70% of the sample points are ”good” borrowers, and 30% are ”bad” which represents a class imbalance issue. An imbalanced class could make it difficult for classification models to learn the minority group and cause bias toward the majority. We will address a solution for this in the following section.

## Cleaning and preparation
The raw data of the German Credit Database required some cleaning and transformations to make it suitable for machine learning models. Our goal was to ensure compatibility with the algorithms while retaining interpretability and maximizing the predictive power of the features.

The first step was selecting only the features relevant to predicting credit risk, such as `age`, `credit_amount`, `savings_status`, and `checking_status`. Selecting a smaller, relevant subset of features reduced noise, improved computational efficiency and made the data more interpretable.

One challenge in the raw dataset was combined variables, such as `personal\_status`, which included both gender and marital status. This column was split into two separate variables, `sex` and `p_status`, allowing independent analysis of their effects on creditworthiness as gender and marital status may influence credit risk differently.

The target variable, which classified individuals as “good” or “bad” credit risks, was transformed into a binary numeric format. Encoding “good” as 0 and “bad” as 1 made it compatible with machine learning models and framed the problem as a binary classification task.

Categorical variables, such as job, `savings_status`, and `checking_status`, were reclassified into simplified categories. For example, job was encoded based on skill level, ranging from unskilled to highly qualified, while account statuses were grouped into levels like “little,” “moderate,” and “rich.” This reclassification reduced the complexity of these features and addressed sparsity issues, where some categories had very few observations. Simplifying these variables also improved interpretability, as the categories became more meaningful and easier to analyze.

A key engineered feature was `repayment_burden`, calculated as the ratio of `credit_amount` to `duration`. This feature captured the financial strain of loan repayment, a potentially strong predictor of credit risk. Additionally, squared versions of numerical features, such as `age` and `credit_amount`, were introduced to allow models to capture non-linear relationships. For instance, the squared terms could highlight situations where risk increases disproportionately with larger loan amounts or longer durations.

To ensure uniformity across variables, all numerical features were normalized to have a mean of 0 and a standard deviation of 1. Normalization was particularly important for distance-based algorithms like K-Nearest Neighbors and for gradient-boosting models that can be sensitive to feature scaling. Without normalization, variables with larger ranges, such as `credit_amount`, could dominate the learning process, leading to biased results. Standardizing the data ensured that all features contributed equally to the model.

Categorical variables were converted into binary, dummy variables, assigning 0 or 1 to each category within a feature. This step was crucial for algorithms that cannot handle non-numeric data, such as logistic regression and XGBoost. Label encoding preserved the distinctiveness of each category while making the dataset fully numeric and compatible with a range of machine learning models.

## Methodology and results
The process of building and evaluating machine learning models for predicting credit risk was structured and systematic, incorporating multiple algorithms to assess their performance. Each model was trained on a balanced data set, created through under-sampling to address the class imbalance. This ensures that the models learn from both classes equally, preventing bias towards the majority class. To prevent over-fitting and provide a reliable assessment of model performance, a 75/25 train-test split was employed. This approach involves training the models on 75\% of the data and evaluating them on the remaining 25\%. Additionally, 7-fold cross-validation was utilized to further enhance model reliability and reduce the impact of data variability. To evaluate the performance of the models, accuracy plots and ROC curves, along with their corresponding AUC scores, will be presented. These metrics provide insights into the models' ability to accurately classify instances and their discrimination power, respectively.

### KNN
The K-Nearest Neighbors (KNN) algorithm is a classification technique that assigns a class label to a new data point based on the majority class of its $k$ nearest neighbors. In the context of credit risk assessment, KNN can be particularly useful. By calculating the distance between a new credit applicant and existing customers, the algorithm identifies the $k$ most similar customers. The majority creditworthiness among these $k$ neighbors is then assigned to the new applicant. This approach leverages the experience with similar customers to inform decisions about new credit applications.

### Random Forest
The Random Forest algorithm is an ensemble method that builds multiple decision trees to improve predictive accuracy. In the implementation, a grid search is used to fine-tune the model by testing combinations of the number of trees (`ntree`) and the number of features per split (`mtry`). Performance is evaluated using out-of-bag (`OOB`) and test accuracy, with the best hyperparameters selected based on the highest test accuracy.

The final model, trained with the optimal ntree and mtry, is evaluated on a test set using metrics like a confusion matrix and ROC curve. This approach ensures a robust and accurate Random Forest model, suitable for tasks such as credit risk assessment, while feature importance enhances interpretability.

### XGBoost
XGBoost is an ensemble learning method that sequentially builds an ensemble of weak learners. In the context of credit risk assessment, XGBoost is effective for binary classification problems. By iteratively adding trees and adjusting their predictions, XGBoost can capture complex relationships between features and the target variable. Additionally, XGBoost is less prone to over-fitting.

To optimize the performance of the XGBoost model, a grid search with 7-fold cross-validation is employed. This involves testing different combinations of hyperparameters, such as the number of boosting rounds, learning rate, tree depth, and regularization parameters. By experimenting with different configurations and evaluating their performance on multiple subsets of the data, the optimal settings can be identified.

### Logit unpenalized and with several penalizations
Several penalized logistic regression models, including ridge, lasso, and elastic net, were implemented to assess credit risk. A grid search with 7-fold cross-validation was used to optimize the alpha parameter, which controls the balance between L1 and L2 regularization. This process identifies the best configuration based on the highest area under the ROC curve (AUC), improving the model's predictive performance while minimizing overfitting.

Ridge and lasso regressions were evaluated alongside elastic net, with coefficients analyzed to understand variable shrinkage. Additionally, an unpenalized logistic regression model was trained for comparison. Model performance was assessed using metrics such as confusion matrices and ROC curves, ensuring robust predictions and interpretability for credit risk assessment.

The above shrinkage plots show the trade-off between model complexity (non-zero coefficients) and goodness-of-fit (binomial deviance). The optimal λ values (lambda.min and lambda.1se) balance this trade-off. Ridge retains all parameters with shrinkage, Elastic Net balances shrinkage and sparsity, and Lasso achieves sparsity by eliminating coefficients. Choosing λ one standard error from the minimum binomial deviance improves interpretability by reducing parameters. For example, Lasso reduces parameters from 27–28 to 14–17, while Elastic Net removes about five. Prioritizing accuracy, we proceed with the parameters at minimum binomial deviance.

### Results - Accuracy
The model comparison plot reveals interesting insights into the performance of various models for credit risk assessment. XGBoost stands out as the top-performing model with the highest accuracy and the tightest confidence interval. This suggests that XGBoost's ability to sequentially build an ensemble of models, combined with its regularization techniques, effectively captures complex patterns in the data and generalizes well to unseen data.

Ensemble methods like Random Forest and XGBoost generally outperform simpler models like Logistic Regression and its regularized variants (Ridge, Lasso, Elastic Net). This highlights the advantage of combining multiple models to improve predictive accuracy and reduce overfitting. However, the performance differences between these models are relatively small, indicating that careful model selection and hyperparameter tuning are crucial to achieve optimal results.

### Results - ROC Curves and AUC

The ROC curve visually compares the performance of various models in distinguishing between positive and negative classes. Random Forest achieves the highest AUC-ROC score (0.813), indicating its superior ability to predict credit risk. XGBoost also performs well, with an AUC of 0.791, showcasing its effectiveness in handling complex patterns. Ridge and Elastic Net models follow closely with AUC values of 0.788, demonstrating strong predictive capabilities. Lasso and standard Logistic Regression models exhibit slightly lower AUC scores but remain competitive, highlighting their utility in specific scenarios.

K-Nearest Neighbors (KNN) has the lowest AUC (0.716), indicating weaker discriminative ability compared to the other methods. Despite this, simpler models like Logistic Regression and its regularized versions may still be preferred in applications where model interpretability or computational efficiency is crucial. Ultimately, the choice of the best model depends on the specific requirements of the credit risk assessment task, such as the need for accuracy, interpretability, or scalability.

## Conclusions
The project focused on predicting credit risk using the German Credit Dataset, using a range of machine learning models to evaluate their effectiveness. Through data cleaning and preparation, the dataset was transformed into a structured, normalized format, ready for analysis. Key steps included selecting relevant features, simplifying categorical variables, engineering new features such as repayment burden, and applying normalization and encoding. These processes ensured the dataset was suitable for modeling.

The analysis implemented and compared six machine learning models: K-Nearest Neighbors (KNN), Random Forest, XGBoost, Ridge Regression, Lasso Regression, and Elastic Net. Each model was optimized using cross-validation and parameter tuning to achieve the best possible performance. The results, visualized in the model comparison graph, demonstrated that ensemble methods like Random Forest and XGBoost consistently outperformed simpler algorithms. XGBoost achieved the highest accuracy, with its ability to handle complex patterns and effective regularization standing out as key strengths. Random Forest followed closely, showcasing its robustness and reliability across multiple tests.

The logistic regression models, particularly Elastic Net, provided a balance between accuracy and interpretability. While their performance was slightly lower than the ensemble methods, they remain valuable for scenarios where transparency is critical. KNN, on the other hand, had the lowest accuracy and highest variability, highlighting its limitations in datasets with high dimensionality and complex relationships.

In conclusion, the comparison of models emphasized that ensemble methods are highly effective for datasets with non-linear relationships and interactions, while simpler models like logistic regression retain their relevance in contexts requiring clear and interpretable outputs.

This project demonstrated how machine learning can improve decision-making in credit risk assessment. By preparing the data and evaluating multiple models, it was possible to identify the most effective approach, providing valuable insights in financial modeling.
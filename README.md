# Foundations of Data Science @ BSE
Foundations of data science final project: Term 1 2024-2025 at Barcelona School of Economics

Names: Jorge Paredes, Catalina Becu, Rebecca Hess, Jorn Diesveld

## Aim of the project:

Predict using diverse methods a factor variable showing if a client is good or bad using the German credit dataset found at: https://www.openml.org/search?type=data&sort=runs&status=active&id=31

## Abstract
This project analyzes credit risk prediction using the German Credit Dataset, classifying individuals as good or bad credit risks based on financial and demographic factors. Seven machine learning models were evaluated: K-Nearest Neighbors (KNN), Random Forest, XGBoost, Logistic Regression, and its variations with Ridge, Lasso, and Elastic Net regularization. The models were compared using accuracy and ROC-AUC metrics. Random Forest performs best, with the highest accuracy and AUC (0.813), making it the most reliable model for distinguishing between “good” and “bad” credit risks. XGBoost and Elastic Net are competitive alternatives, while K-Nearest Neighbors shows the weakest performance (AUC 0.716). These findings highlight the importance of model evaluation in credit risk prediction and demonstrate that nonlinear approaches consistently outperform linear models in Supervised Learning for prediction. This analysis provides valuable insights into how machine learning can improve decision-making in financial contexts.

```
Results to be found in RMD or Jupyter notebooks in code section
```

## Conclusions
The project focused on predicting credit risk using the German Credit Dataset, using a range of machine learning models to evaluate their effectiveness. Through data cleaning and preparation, the dataset was transformed into a structured, normalized format, ready for analysis. Key steps included selecting relevant features, simplifying categorical variables, engineering new features such as repayment burden, and applying normalization and encoding. These processes ensured the dataset was suitable for modeling.

The analysis implemented and compared six machine learning models: K-Nearest Neighbors (KNN), Random Forest, XGBoost, Ridge Regression, Lasso Regression, and Elastic Net. Each model was optimized using cross-validation and parameter tuning to achieve the best possible performance. The results, visualized in the model comparison graph, demonstrated that ensemble methods like Random Forest and XGBoost consistently outperformed simpler algorithms. XGBoost achieved the highest accuracy, with its ability to handle complex patterns and effective regularization standing out as key strengths. Random Forest followed closely, showcasing its robustness and reliability across multiple tests.

The logistic regression models, particularly Elastic Net, provided a balance between accuracy and interpretability. While their performance was slightly lower than the ensemble methods, they remain valuable for scenarios where transparency is critical. KNN, on the other hand, had the lowest accuracy and highest variability, highlighting its limitations in datasets with high dimensionality and complex relationships.

In conclusion, the comparison of models emphasized that ensemble methods are highly effective for datasets with non-linear relationships and interactions, while simpler models like logistic regression retain their relevance in contexts requiring clear and interpretable outputs.

This project demonstrated how machine learning can improve decision-making in credit risk assessment. By preparing the data and evaluating multiple models, it was possible to identify the most effective approach, providing valuable insights in financial modeling.
# Wine Quality Machine Learning Project

This project explores the **Wine Quality** dataset using various data science techniques, from **descriptive statistics** and **hypothesis testing** to **machine learning models** including classification, clustering, and hyperparameter tuning.

---

##  Google Colab Notebook

[📘 Open Full Colab Notebook](https://colab.research.google.com/drive/1-3Y31sEmkWcxognQBQiZwcks8h_FPHZM?usp=sharing)


---

##  Dataset

We used the **`winequality-red.csv`** dataset, which contains physicochemical tests and quality ratings for red wine samples.

---

##  Project Breakdown

# 1. Descriptive & Inferential Statistics
- Computed: mean, median, mode, variance, std dev, skewness, and kurtosis.
- Visualized with histograms and boxplots.
- Hypothesis: Higher alcohol content leads to better wine quality.
- Performed a t-test and computed a 95% confidence interval.

# 2. Correlation & Regression
- Correlation heatmap for all features.
- Simple linear regression (1 predictor).
- Multiple regression with 3+ predictors.
- Evaluated using R², MAE, MSE, RMSE.

# 3. Machine Learning Theory
- Explained Supervised vs Unsupervised learning.
- Discussed Bias-Variance tradeoff.
- Compared Accuracy, Precision, Recall, and F1-score.

# 4. Classification Models
- Models used: Logistic Regression, Decision Tree, Random Forest, SVM.
- Evaluated with:
  - Confusion Matrix
  - Classification Report
  - ROC Curve
- **Best Model**: Random Forest with ROC-AUC ~0.87

# 5. Clustering (Unsupervised Learning)
- K-Means: Chose optimal K using silhouette score.
- DBSCAN: Tuned `eps` and `min_samples`.
- Visualized clusters using PCA.

# 6. Feature Engineering & Dimensionality Reduction
- Handled missing values using mean, median.
- Created new features.
- Applied StandardScaler and MinMaxScaler.
- Used PCA and t-SNE for dimensionality reduction and visualization.

# 7. Hyperparameter Tuning
- Used GridSearchCV and RandomizedSearchCV on Random Forest.
- Best Parameters: `{ 'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200 }`
- Final ROC-AUC Score: **0.79**

---

##  Summary & Reflection

This project gave me hands-on experience with the **full data science pipeline**:
- From exploring and cleaning data,
- To building and tuning powerful machine learning models.
I now feel more confident in using tools like **pandas, scikit-learn, and matplotlib**, and better understand the **importance of evaluation metrics** and model optimization in real-world problems.

---

##  Author

**GitHub**: [@Busoki](https://github.com/Busoki)  
**Notebook by**: Okirya




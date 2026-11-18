# Linear Regression Practicals

This directory contains complete, hands-on implementations and practical notebooks for **Linear Regression**, covering fundamental concepts, mathematics, assumptions, diagnostics, and real-world datasets.

---

## 📑 Contents

| File | Description | Dataset Source | Target Variable |
| :--- | :--- | :--- | :--- |
| `Linear_Regression_Real_World_Practical.ipynb` | Comprehensive real-world Multiple Linear Regression with full EDA, assumptions checking, and evaluation | California Housing | `MedHouseVal` |
| `insurance.ipynb` | Medical insurance cost prediction with categorical feature encoding and multi-feature regression | `insurance.csv` | `charges` |
| `house_pricing_boston.ipynb` | Simple linear regression baseline exploring study hours vs. exam score relationship | Embedded | `Exam_score` |
| `heart.ipynb` | Exploratory data analysis, correlation diagnostics, and clinical metrics modeling | `heart.csv` | Clinical features |
| `insurance.csv` | Medical insurance records (1,338 samples, 7 features) | Kaggle | N/A |
| `heart.csv` | Cardiovascular disease diagnostic dataset (918 samples, 12 features) | Kaggle / UCI | N/A |

---

## 🧠 Theory & Mathematical Foundations

### 1. Simple Linear Regression
Simple Linear Regression models the relationship between a single independent variable $X$ and a dependent variable $Y$:
$$Y = \beta_0 + \beta_1 X + \varepsilon$$

Where:
- $\beta_0$: Intercept (value of $Y$ when $X = 0$)
- $\beta_1$: Slope coefficient (change in $Y$ per unit change in $X$)
- $\varepsilon$: Error term ($\varepsilon \sim \mathcal{N}(0, \sigma^2)$)

### 2. Multiple Linear Regression
Models the relationship between multiple features $X_1, X_2, \dots, X_p$ and target $Y$:
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \varepsilon = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

### 3. Cost Function: Mean Squared Error (MSE)
We seek coefficients $\boldsymbol{\beta}$ that minimize the sum of squared residuals:
$$J(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2 = \frac{1}{n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$$

### 4. Closed-Form Solution: Normal Equation (OLS)
Setting $\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = 0$ yields:
$$\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

---

## 🔍 Key Assumptions of Linear Regression

1. **Linearity**: The relationship between features and the target variable is linear.
2. **Independence**: Observations are independent of each other (no autocorrelation in residuals).
3. **Homoscedasticity**: The residuals have constant variance across all levels of the independent variables.
4. **Normality of Residuals**: The residual errors $\varepsilon$ are normally distributed.
5. **No Multicollinearity**: Independent variables are not highly correlated with each other ($VIF < 5$).

---

## 📊 Evaluation Metrics

- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Mean Squared Error (MSE)**:
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\text{MSE}}$$
- **Coefficient of Determination ($R^2$ Score)**:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## 🚀 How to Run the Notebooks

1. Ensure the parent repository's dependencies are installed:
   ```bash
   pip install -r ../../requirements.txt
   ```
2. Launch Jupyter inside this directory or from the root:
   ```bash
   jupyter lab
   ```
3. Open any `.ipynb` notebook and execute cells sequentially.


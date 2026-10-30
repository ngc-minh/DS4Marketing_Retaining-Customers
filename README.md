# 📉 Telco Customer Churn: Data-centric vs Model-centric

Predicting which telecom customers are about to leave, and comparing two ways to improve a model: **improving the data** (feature selection and rebalancing) versus **tuning the models**.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

## Dataset

IBM **Telco Customer Churn** sample: **7,043 customers**, 21 columns (demographics, services, contract, billing and the `Churn` label). Included in [`data/`](data/).

## Two approaches

| | Data-centric ([`Data-centrix/`](Data-centrix/)) | Model-centric ([`Model-centrix/`](Model-centrix/)) |
|---|---|---|
| Idea | Keep models simple, improve the data | Keep the data as is, tune the models |
| Steps | EDA, missing values, encoding, scaling, **chi-square** test for categorical features, **ANOVA** for numerical features, **SMOTE** to balance classes | Stratified split, hyperparameter search optimized for F1 |
| Models | Random Forest, Gradient Boosting, Logistic Regression, MLP | Random Forest, Gradient Boosting, Logistic Regression, ANN (one notebook each) |

## Results

F1-score for the **churn** class on the test split:

| Model | Model-centric | Data-centric |
|---|---|---|
| Gradient Boosting | 0.60 | **0.84** |
| Random Forest | 0.59 | 0.79 |
| Logistic Regression | 0.61 | 0.78 |
| Neural network | 0.58 | 0.78 |

Overall accuracy is around 0.80 for the model-centric models and 0.77–0.84 for the data-centric ones.

## Limitations and next steps

- In the data-centric notebook, SMOTE is applied **before** the train/test split, so its test set is balanced and contains synthetic customers. The two columns above are therefore not directly comparable, and the data-centric scores are optimistic.
- A fair comparison resamples only the training data and evaluates both approaches on the same untouched test set.

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter notebook
```

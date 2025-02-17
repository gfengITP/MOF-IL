import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

file_path = 'MOF-IL.csv'
data = pd.read_csv(file_path)

X = data[['PLD', 'LCD', 'SSA', 'porosity', 'density', 'pore_dimension']]
y = data['capa_mass'] # or 'capa_volume' '095_current' For volumetric capcitance, separate training and testing should be conducted for materials with 1D and 3D pores 

label_encoder = LabelEncoder()
X['pore_dimension'] = label_encoder.fit_transform(X['pore_dimension'])

# Stratified sampling
percentiles = np.percentile(y, np.arange(0, 101, 10))
y_strata = np.digitize(y, percentiles[1:-1]) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_strata)

# Optuna optimization
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
    }
    
    # 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = []
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = XGBRegressor(**param, n_jobs=7)
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        mse = mean_squared_error(y_val_fold, preds)
        mse_scores.append(mse)
    
    return np.mean(mse_scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)
best_params = study.best_params

#Train and Evaluate the model on the test set
model = XGBRegressor(**best_params, n_jobs=7)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Test Set Performance:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Figure
plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel("MD calcualtion")
plt.ylabel("ML prediction")
plt.show()

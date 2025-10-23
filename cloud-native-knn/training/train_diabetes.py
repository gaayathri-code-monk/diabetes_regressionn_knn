from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from pathlib import Path
import json

RANDOM_STATE = 42

def eval_reg(y_true, y_pred):
    return {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred))
    }

def main():
    data = load_diabetes()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
    param_grid = {'knn__n_neighbors':[3,5,7,9,11],'knn__weights':['uniform','distance'],'knn__p':[1,2]}
    knn_search = GridSearchCV(knn_pipe, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    knn_search.fit(X_train, y_train)
    knn_best = knn_search.best_estimator_

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    y_pred_knn = knn_best.predict(X_test)
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    knn_metrics = eval_reg(y_test, y_pred_knn)
    lr_metrics = eval_reg(y_test, y_pred_lr)
    rf_metrics = eval_reg(y_test, y_pred_rf)

    metrics = {
        'task':'regression',
        'dataset':'diabetes',
        'best_params': knn_search.best_params_,
        **knn_metrics,
        'comparisons': {
            'KNN(best)': knn_metrics,
            'LinearRegression': lr_metrics,
            'RandomForest': rf_metrics
        }
    }

    Path('app/models').mkdir(parents=True, exist_ok=True)
    dump(knn_best, 'app/models/diabetes_knn.joblib')
    dump(lr, 'app/models/diabetes_linreg.joblib')
    dump(rf, 'app/models/diabetes_rf.joblib')

    Path('training/diabetes_metrics.json').write_text(json.dumps(metrics, indent=2))

    print('Saved: app/models/diabetes_knn.joblib, diabetes_linreg.joblib, diabetes_rf.joblib')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()

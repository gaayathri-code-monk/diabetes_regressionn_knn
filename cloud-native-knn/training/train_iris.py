from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

RANDOM_STATE = 42

def eval_cls(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro'))
    }

def save_cm_png(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    labels = iris.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # KNN with GridSearch
    knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
    param_grid = {'knn__n_neighbors':[3,5,7,9],'knn__weights':['uniform','distance'],'knn__p':[1,2]}
    knn_search = GridSearchCV(knn_pipe, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    knn_search.fit(X_train, y_train)
    knn_best = knn_search.best_estimator_

    # Baseline/alt models
    logreg = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
    logreg.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred_knn = knn_best.predict(X_test)
    y_pred_lr = logreg.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_knn)

    metrics = {
        'task':'classification',
        'dataset':'iris',
        'best_params': knn_search.best_params_,
        **eval_cls(y_test, y_pred_knn),
        'confusion_matrix': cm.tolist(),
        'comparisons': {
            'KNN(best)': eval_cls(y_test, y_pred_knn),
            'LogisticRegression': eval_cls(y_test, y_pred_lr),
            'RandomForest': eval_cls(y_test, y_pred_rf)
        }
    }

    # Save models
    Path('app/models').mkdir(parents=True, exist_ok=True)
    dump(knn_best, 'app/models/iris_knn.joblib')
    dump(logreg, 'app/models/iris_logreg.joblib')
    dump(rf, 'app/models/iris_rf.joblib')

    # Save confusion matrix image
    save_cm_png(cm, labels, 'app/static/iris_cm.png')

    # Save metrics
    Path('training').mkdir(exist_ok=True)
    Path('training/iris_metrics.json').write_text(json.dumps(metrics, indent=2))

    print('Saved: app/models/iris_knn.joblib, iris_logreg.joblib, iris_rf.joblib')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()

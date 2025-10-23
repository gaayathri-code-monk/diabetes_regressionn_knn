import json
from pathlib import Path
import pandas as pd

def main():
    iris = {}
    diabetes = {}

    if Path('training/iris_metrics.json').exists():
        iris = json.loads(Path('training/iris_metrics.json').read_text())
    if Path('training/diabetes_metrics.json').exists():
        diabetes = json.loads(Path('training/diabetes_metrics.json').read_text())

    metrics = {'iris': iris, 'diabetes': diabetes}
    Path('training/metrics.json').write_text(json.dumps(metrics, indent=2))

    # Export CSVs
    out_dir = Path('training')
    out_dir.mkdir(exist_ok=True)
    # Iris metrics
    if iris:
        iris_scores = {k: iris.get(k) for k in ['accuracy','precision_macro','recall_macro','f1_macro']}
        pd.DataFrame([iris_scores]).to_csv(out_dir / 'iris_metrics.csv', index=False)
        if 'comparisons' in iris:
            pd.DataFrame(iris['comparisons']).T.to_csv(out_dir / 'iris_model_comparisons.csv')
    # Diabetes metrics
    if diabetes:
        db_scores = {k: diabetes.get(k) for k in ['mse','mae','r2']}
        pd.DataFrame([db_scores]).to_csv(out_dir / 'diabetes_metrics.csv', index=False)
        if 'comparisons' in diabetes:
            pd.DataFrame(diabetes['comparisons']).T.to_csv(out_dir / 'diabetes_model_comparisons.csv')

    # Combined summary
    combined = []
    if iris:
        row = {'task':'iris', **{k: iris.get(k) for k in ['accuracy','precision_macro','recall_macro','f1_macro']}}
        combined.append(row)
    if diabetes:
        row = {'task':'diabetes', **{k: diabetes.get(k) for k in ['mse','mae','r2']}}
        combined.append(row)
    if combined:
        pd.DataFrame(combined).to_csv(out_dir / 'metrics_summary.csv', index=False)

    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()

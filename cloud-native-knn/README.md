

---

## 🆕 v1.4.0 — Runtime Model Toggle + CSV Metrics + GHCR Publish

### Runtime model selection
- **Iris**: `model=knn|logreg|rf`
- **Diabetes**: `model=knn|linreg|rf`
- Works via UI dropdowns or query params on `/predict/...`.
- Additional models are trained & saved during `make train`:
  - `app/models/iris_logreg.joblib`, `app/models/iris_rf.joblib`
  - `app/models/diabetes_linreg.joblib`, `app/models/diabetes_rf.joblib`

### CSV exports
- After `make evaluate`, CSVs appear in `training/`:
  - `iris_metrics.csv`, `iris_model_comparisons.csv`
  - `diabetes_metrics.csv`, `diabetes_model_comparisons.csv`
  - `metrics_summary.csv`

### GHCR publish workflow
- Workflow: `.github/workflows/ghcr-publish.yml`
- Triggers on tags like `v1.4.0`
- Requires repo secrets:
  - `GHCR_TOKEN` → a Personal Access Token with `write:packages`
  - (Optional) `GHCR_USERNAME` → defaults to `github.repository_owner` if unset
- Pushes images to: `ghcr.io/<owner>/cloud-native-knn:<tag>` and `:latest`

### Example
```bash
git tag v1.4.0
git push origin v1.4.0
# -> GitHub Actions builds and pushes image to GHCR
```


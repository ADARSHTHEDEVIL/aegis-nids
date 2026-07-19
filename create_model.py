from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import numpy as np

# Create a simple trained-like RandomForest model
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Fit it with dummy data so predict_proba() works
X_dummy = np.random.randn(100, 13)  # 13 features
y_dummy = np.random.randint(0, 2, 100)
model.fit(X_dummy, y_dummy)

# Save it
registry_dir = Path('src/models/registry')
registry_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, registry_dir / 'nids_model.joblib')
print("✅ Model created with predict_proba()!")

from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import numpy as np

# Create model with CORRECT feature count (8, matching your preprocessor)
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Fit with 8 features (matching preprocessor output)
X_dummy = np.random.randn(100, 8)  # 8 features, not 13!
y_dummy = np.random.randint(0, 2, 100)
model.fit(X_dummy, y_dummy)

registry_dir = Path('src/models/registry')
registry_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, registry_dir / 'nids_model.joblib')
print("✅ Model created with 8 features!")

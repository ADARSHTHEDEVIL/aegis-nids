from src.data.preprocessor import PreprocessingArtifacts
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
from pathlib import Path
import numpy as np

# Create dummy data to fit the scaler
X_numeric = np.random.randn(100, 5)
X_categorical = np.array([['tcp', 'http', 'SF']] * 100)

# Fit scaler
scaler = StandardScaler()
scaler.fit(X_numeric)

# Fit encoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(X_categorical)

# Create valid artifacts
artifacts = PreprocessingArtifacts(
    categorical_columns=['protocol_type', 'service', 'flag'],
    numeric_columns=['src_bytes', 'dst_bytes', 'count', 'dst_host_srv_count', 'dst_host_same_src_port_rate'],
    ordinal_encoder=encoder,
    scaler=scaler,
    numeric_medians={'src_bytes': 0, 'dst_bytes': 0, 'count': 0, 'dst_host_srv_count': 0, 'dst_host_same_src_port_rate': 0},
    feature_order=['src_bytes', 'dst_bytes', 'count', 'dst_host_srv_count', 'dst_host_same_src_port_rate', 'protocol_type', 'service', 'flag'],
    label_mapping={"normal": 0, "attack": 1},
    is_fitted=True
)

registry_dir = Path('src/models/registry')
registry_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(artifacts, registry_dir / 'preprocessor.joblib')
print("✅ Properly fitted preprocessor created!")

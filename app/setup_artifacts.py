from src.data.preprocessor import PreprocessingArtifacts
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
from pathlib import Path

# Create valid PreprocessingArtifacts
artifacts = PreprocessingArtifacts(
    categorical_columns=['protocol_type', 'service', 'flag'],
    numeric_columns=['src_bytes', 'dst_bytes', 'count', 'dst_host_srv_count', 'dst_host_same_src_port_rate'],
    ordinal_encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    scaler=StandardScaler(),
    numeric_medians={},
    feature_order=['src_bytes', 'dst_bytes', 'count', 'protocol_type', 'service', 'flag'],
    label_mapping={"normal": 0, "attack": 1},
    is_fitted=True
)

# Save it
registry_dir = Path('src/models/registry')
registry_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(artifacts, registry_dir / 'preprocessor.joblib')
print("✅ Done!")
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_dir : Path
    all_schema: Path
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path    
    
    


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir : Path
    X_train_data_path: Path
    X_test_data_path: Path
    y_train_data_path: Path
    y_test_data_path: Path
    model_name : str
    alpha: float
    n_estimators: int
    max_depth: int
    learning_rate: float
    

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    X_test_data_path: Path
    y_test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_path: Path
    target_column: str
    mlflow_uri: str
    
from ml_project.constants import *
from ml_project.utils.common import read_yaml , create_directories
from ml_project.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelEvaluationConfig, ModelTrainingConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_data_validation_config(self)-> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])
        
        return DataValidationConfig(
            root_dir = config.root_dir,
            STATUS_FILE= config.STATUS_FILE,
            unzip_dir=config.unzip_dir,
            all_schema=schema
        )
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    
    
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        params = self.params.LGBM
        schema = self.schema.TARGET
        
        model_training_config = ModelTrainingConfig(
            root_dir= Path(config.root_dir),
            X_train_data_path= Path(config.X_train_data_path),
            X_test_data_path= Path(config.X_test_data_path),
            y_train_data_path= Path(config.y_train_data_path),
            y_test_data_path= Path(config.y_test_data_path),
            model_name= config.model_name,
            alpha= params.alpha,
            n_estimators= params.n_estimators,
            max_depth= params.max_depth,
            learning_rate= params.learning_rate
        ) 
        
        return model_training_config
    
    
    def __init__(self, config_filepath= CONFIG_FILE_PATH , params_filepath=PARAMS_FILE_PATH ,schema_filepath=SCHEMA_FILE_PATH ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([self.config.model_training.root_dir])
        
        
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        params = self.params.LGBM
        schema = self.schema.TARGET
        
        model_training_config = ModelTrainingConfig(
            root_dir= Path(config.root_dir),
            X_train_data_path= Path(config.X_train_data_path),
            X_test_data_path= Path(config.X_test_data_path),
            y_train_data_path= Path(config.y_train_data_path),
            y_test_data_path= Path(config.y_test_data_path),
            model_name= config.model_name,
            alpha= params.alpha,
            n_estimators= params.n_estimators,
            max_depth= params.max_depth,
            learning_rate= params.learning_rate
        ) 
        
        return model_training_config    
    
    
    def __init__(self, config_filepath= CONFIG_FILE_PATH , params_filepath=PARAMS_FILE_PATH ,schema_filepath=SCHEMA_FILE_PATH ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([self.config.artifacts_root])
        
        
    def get_model_evaluation_config(self):
        config = self.config.model_evaluation
        params = self.params.LGBM
        schema = self.schema.TARGET
        
        create_directories([config.root_dir])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            X_test_data_path=Path(config.X_test_data_path),
            y_test_data_path=Path(config.y_test_data_path),
            model_path=Path(config.model_path),
            all_params=params,
            metric_file_path=Path(config.metric_file_path),
            target_column=schema.y,
            mlflow_uri=config.mlflow_uri
            
        )
        
        return model_evaluation_config
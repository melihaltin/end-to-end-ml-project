import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,accuracy_score, r2_score
from ml_project.entity.config_entity import ModelEvaluationConfig
import pandas as pd
import joblib
import mlflow
from urllib.parse import urlparse
from ml_project.utils.common import save_json


class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        acc = accuracy_score(actual, pred)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2, acc
    
    def log_into_mlflow(self, nested=False):
        X_test = pd.read_csv(self.config.X_test_data_path)
        y_test = pd.read_csv(self.config.y_test_data_path)
        
        model = joblib.load(self.config.model_path)
        
      
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run() :
            prediction_qualities = model.predict(X_test)
            (rmse, mae, r2 , acc) = self.eval_metrics(y_test, prediction_qualities)
            
            scores = {
                "acc": acc,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            
            save_json(self.config.metric_file_path, data = scores)
            
            mlflow.log_params(self.config.all_params)
            
            
            mlflow.log_metrics(scores)
            # mlflow.log_metrics('mae', mae)
            # mlflow.log_metrics('r2', r2)
            
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model" , registered_model_name="LGBM")
                
            else:
                mlflow.sklearn.log_model(model, "model")        
import os
from turtle import pd
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from ml_project.entity.config_entity import ModelTrainingConfig


class ModelTrainer:
    def __init__(self , config: ModelTrainingConfig) -> None:
        self.config = config

    def train(self):
        X_train = pd.read_csv(self.config.X_train_data_path)
        X_test = pd.read_csv(self.config.X_test_data_path)
        y_train = pd.read_csv(self.config.y_train_data_path)
        y_test = pd.read_csv(self.config.y_test_data_path)
    
        
        clf = LGBMClassifier(alpha=self.config.alpha,
                             n_estimators=self.config.n_estimators,
                             max_depth=self.config.max_depth,
                             learning_rate=self.config.learning_rate)
        
        clf.fit(X_train, y_train)
        
        joblib.dump(clf, os.path.join(self.config.root_dir, f"{self.config.model_name}.joblib"))
        
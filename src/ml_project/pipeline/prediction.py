import joblib
import pandas as pd
import numpy as np
from pathlib import Path


class PredictionPipeline:
    def __init__(self, model_path: Path):
        self.model = joblib.load(Path('artifacts/model_trainer/LGBM.joblib'))
        
        
    def predict(self, input_data: pd.DataFrame) -> np.array:
        return self.model.predict(input_data)    
        
        
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import LabelEncoder,StandardScaler

class DataTransformation:

    def __init__(self , df):
        self.df = df

    def binary_encoding(self, positive_value="yes"):
        columns = ['default', 'housing', 'loan']
        for col in columns:
            print(col)
            self.df[col] = self.df[col].apply(lambda x: 1 if x == positive_value else 0)
        

    def month_transformer(self) -> pd.DataFrame:
        # print(self.df.columns)
        month_mapping = {
            'jan': 1,
            'feb': 2,
            'mar': 3,
            'apr': 4,
            'may': 5,
            'jun': 6,
            'jul': 7,
            'aug': 8,
            'sep': 9,
            'oct': 10,
            'nov': 11,
            'dec': 12
        }
        self.df['month'] = self.df['month'].str.lower()
        self.df.month = self.df.month.map(month_mapping)

        self.df.day = self.df.day.astype(str).str.zfill(2)
        self.df.month = self.df.month.astype(str).str.zfill(2)

        self.df['date'] = self.df.day + '-' + self.df.month

        self.df['date_month'] = pd.to_datetime('2024-' + self.df['date'], format='%Y-%d-%m', errors='coerce')\
                            .dt.strftime('%d-%m')

        self.df.drop(['date', 'day', 'month'], axis=1, inplace=True)
        
        
        
        
    def age_transformer(self) -> pd.DataFrame:
        blanks = []
        
        for age in self.df['age']:
            if 18 <= age <= 30:
                blanks.append('18-30')
            elif 31 <= age <= 40:
                blanks.append('31-40')
            elif 41 <= age <= 50:
                blanks.append('41-50')
            elif 51 <= age <= 60:
                blanks.append('51-60')
            elif 61 <= age <= 70:
                blanks.append('61-70')
            elif 71 <= age <= 80:
                blanks.append('71-80')
            elif 81 <= age <= 90:
                blanks.append('81-90')
            elif 91 <= age <= 95:
                blanks.append('91-95')
            else:
                blanks.append('Unknown')
                
        self.df['age_group'] = blanks   
        self.df.drop('age', axis=1, inplace=True)
         

    
    def categorical_encoding(self):
        
        one_hot_columns = ['job', 'marital', 'education', 'contact', 'poutcome']
        label_columns = ['age_group' , 'date_month'] 
        
        
        test = {'job_blue-collar': 0, 'job_entrepreneur': 0, 'job_housemaid': 0, 'job_management': 0, 'job_retired': 0, 'job_self-employed': 0, 'job_services': 0, 'job_student': 0, 'job_technician': 0, 'job_unemployed': 0, 'job_unknown': 0, 'marital_married': 0, 'marital_single': 0, 'education_secondary': 0, 'education_tertiary': 0, 'education_unknown': 0, 'contact_telephone': 0, 'contact_unknown': 0, 'poutcome_other': 0, 'poutcome_success': 0, 'poutcome_unknown': 0}
        
        test = pd.DataFrame(test, index=[0])
        self.df = pd.concat([self.df, test], axis=1)
        
        

        for col in one_hot_columns:
            a = self.df[col][0]
            a = col + '_' + a
            if a in self.df.columns:
                self.df[a] = 1
                
        self.df.drop(one_hot_columns, axis=1, inplace=True)
    
    
        le = LabelEncoder()
        for col in label_columns:
            self.df[col] = le.fit_transform(self.df[col])
    
    
    def scaling(self):
        scaler = StandardScaler()
        continuous_features = self.df[[col for col in self.df.columns if self.df[col].dtype != 'object']]
        self.df[continuous_features.columns] = scaler.fit_transform(continuous_features)
    

    def transform(self):
        print("month transformer")
        self.month_transformer()
        print("age transformer")
        self.age_transformer()

        print("scaling")
        self.scaling()
        print("binary")
        self.binary_encoding()
        print("categorical encoding")
        self.categorical_encoding()
        print("return")
        
        print(self.df.columns)
        
        
        return self.df

class   PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_training/LGBM.joblib'))
        
        
        
    def predict(self, input_data: pd.DataFrame):
        transformation = DataTransformation(input_data)  
        input_data = transformation.transform()
        print(self.model.n_features_in_)
        
        print(input_data.shape)
        return self.model.predict(input_data)    
        
        
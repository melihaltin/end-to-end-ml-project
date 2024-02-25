import os
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from ml_project import logger
from ml_project.entity.config_entity import DataTransformationConfig

class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.df = pd.read_csv(self.config.data_path , sep=';')

    def binary_encoding(df, columns, positive_value):
        for col in columns:
            df[col] = df[col].apply(lambda x: 1 if x == positive_value else 0)
        return df     

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
        
        
    
        

    def binary(self):
        def binary_encoding(df, columns, positive_value):
            for col in columns:
                df[col] = df[col].apply(lambda x: 1 if x == positive_value else 0)
            return df 
        binary_columns = ['default', 'housing', 'loan', 'y']
        self.df = binary_encoding(self.df, binary_columns, 'yes')
        
    
    def categorical_encoding(self):
        
        one_hot_columns = ['job', 'marital', 'education', 'contact', 'poutcome']
        label_columns = ['age_group' , 'date_month'] 
        
        self.df = pd.get_dummies(self.df, columns=one_hot_columns, drop_first=True ,  dtype='int64' )
        
        le = LabelEncoder()
        for col in label_columns:
            self.df[col] = le.fit_transform(self.df[col])
    
    
    def scaling(self):
        scaler = StandardScaler()
        continuous_features = self.df[[col for col in self.df.columns if self.df[col].dtype != 'object']]
        self.df[continuous_features.columns] = scaler.fit_transform(continuous_features)
    
    
    
    def train_test_split(self):
        X = self.df.drop('y', axis=1)
        y = self.df['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Splited data into training and test sets")
        logger.info(X_train.shape)
        logger.info(y_train.shape)
        X_train.to_csv(os.path.join(self.config.root_dir, "X_train.csv"),index = False)
        X_test.to_csv(os.path.join(self.config.root_dir, "X_test.csv"),index = False)
        y_train.to_csv(os.path.join(self.config.root_dir, "y_train.csv"),index = False)
        y_test.to_csv(os.path.join(self.config.root_dir, "y_test.csv"),index = False)

        return X_train, X_test, y_train, y_test
    
    def transform(self):
        self.month_transformer()
        self.age_transformer()
        self.scaling()
        self.binary()
        self.categorical_encoding()
        self.train_test_split()
        
            
        
    
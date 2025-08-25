import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


class CustomException(Exception):
    def __init__(self, message, sys):
        super().__init__(message)
        self.message = message

class logging:
    @staticmethod
    def info(message):
        print(f"[INFO] {message}")
    @staticmethod
    def warning(message):
        print(f"[WARNING] {message}")
        
def save_object(file_path, obj):
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


class DataTransformation:
    def __init__(self):
        
        self.preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')

    def get_data_transformer_object(self, numerical_cols, categorical_cols):
        
        try:
            logging.info("Creating data transformer object.")
            
            # Create pipelines for numerical and categorical data
            numerical_pipeline = Pipeline(steps=[
                
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info(f"Numerical columns: {numerical_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")

            
            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_cols),
                    ('categorical_pipeline', categorical_pipeline, categorical_cols)
                ],
                remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, data_path):
        
        try:
            logging.info("Starting data transformation process for 'hotel_booking.csv'.")
            
            
            logging.info(f"Reading data from {data_path}")
            df = pd.read_csv(data_path)


            logging.info("Cleaning data and engineering new features.")

            
            cols_to_drop = [
                'name', 'email', 'phone-number', 'credit_card', 
                'reservation_status', 'reservation_status_date',
                'agent', 'company',
            ]
            df = df.drop(columns=cols_to_drop)
            
            # Drop rows with missing values in key columns
            df.dropna(subset=['country'], inplace=True)
            
            # Handle rows where guests are zero
            df = df[(df['adults'] > 0) | (df['children'] > 0) | (df['babies'] > 0)]

            # Impute missing 'adr' values with the mean
            df['adr'] = df['adr'].replace(0, df['adr'].mean())

            # Create a new feature for the total number of nights stayed
            df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
            
            # Drop the original night columns as they are now combined
            df = df.drop(columns=['stays_in_weekend_nights', 'stays_in_week_nights'])

            
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include='object').columns.tolist()

            
            preprocessor = self.get_data_transformer_object(numerical_cols, categorical_cols)
            
            logging.info("Applying preprocessing object on the dataset.")
            
            transformed_data = preprocessor.fit_transform(df)

            
            save_object(self.preprocessor_obj_path, preprocessor)
            
            logging.info(f"Saved preprocessing object at {self.preprocessor_obj_path}")

            return transformed_data, self.preprocessor_obj_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    
    data_transformer = DataTransformation()
    
    
    file_path = r"C:\Users\uers\Documents\model_building_excercises\hotel booking customers\data\hotel_booking.csv"
    
    transformed_array, preprocessor_path = data_transformer.initiate_data_transformation(file_path)
    
    logging.info(f"Data transformation complete. Transformed data shape: {transformed_array.shape}")
    logging.info(f"Preprocessor object saved at: {preprocessor_path}")

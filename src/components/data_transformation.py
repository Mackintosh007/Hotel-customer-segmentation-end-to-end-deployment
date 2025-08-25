import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Placeholder for custom modules to make the script runnable as a standalone file
# In a real project, you would uncomment the following lines and import from your src directory.
# from src.exception import CustomException
# from src.logger import logging
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
    """
    Saves a Python object to a file using pickle.
    """
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
        """
        Initializes the DataTransformation class with a path to save preprocessor objects.
        """
        # Define the directory where preprocessing artifacts will be saved
        self.preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')

    def get_data_transformer_object(self, numerical_cols, categorical_cols):
        """
        Creates and returns a preprocessor object that combines different
        transformations for numerical and categorical features.
        
        Args:
            numerical_cols (list): A list of column names for numerical features.
            categorical_cols (list): A list of column names for categorical features.
        
        Returns:
            ColumnTransformer: A preprocessor object ready to be fitted and transformed.
        """
        try:
            logging.info("Creating data transformer object.")
            
            # Create pipelines for numerical and categorical data
            numerical_pipeline = Pipeline(steps=[
                # Use StandardScaler to standardize numerical features. This is crucial for K-Means.
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                # Use OneHotEncoder to convert categorical features into a numerical format.
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info(f"Numerical columns: {numerical_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")

            # Combine the pipelines into a single preprocessor object
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
        """
        Orchestrates the data transformation process for the hotel_booking.csv dataset.
        
        Args:
            data_path (str): The file path to the raw data.
        
        Returns:
            tuple: A tuple containing the transformed data (as a NumPy array) and the
                   path to the saved preprocessor object.
        """
        try:
            logging.info("Starting data transformation process for 'hotel_booking.csv'.")
            
            # --- Step 1: Data Loading ---
            logging.info(f"Reading data from {data_path}")
            df = pd.read_csv(data_path)

            # --- Step 2: Data Cleaning and Feature Engineering ---
            logging.info("Cleaning data and engineering new features.")

            # Identify columns to drop (personal identifiers and irrelevant data)
            cols_to_drop = [
                'name', 'email', 'phone-number', 'credit_card', 
                'reservation_status', 'reservation_status_date',
                'agent', 'company',
            ]
            df = df.drop(columns=cols_to_drop)
            
            # Handle rows where guests are zero
            df = df[(df['adults'] > 0) | (df['children'] > 0) | (df['babies'] > 0)]

            # Impute missing 'adr' values with the mean
            df['adr'] = df['adr'].replace(0, df['adr'].mean())
            
            # Drop all remaining rows with any missing values
            df.dropna(inplace=True)

            # Create a new feature for the total number of nights stayed
            df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
            
            # Drop the original night columns as they are now combined
            df = df.drop(columns=['stays_in_weekend_nights', 'stays_in_week_nights'])

            # --- Step 3: Identify Features to Transform ---
            # Automatically identify numerical and categorical features from the cleaned DataFrame
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include='object').columns.tolist()

            # --- Step 4: Get and Fit the Preprocessor ---
            preprocessor = self.get_data_transformer_object(numerical_cols, categorical_cols)
            
            logging.info("Applying preprocessing object on the dataset.")
            # Fit and transform the data in one step
            transformed_data = preprocessor.fit_transform(df)

            # --- Step 5: Save the Preprocessor Object ---
            # It's crucial to save the preprocessor to use it on new, unseen data later.
            save_object(self.preprocessor_obj_path, preprocessor)
            
            logging.info(f"Saved preprocessing object at {self.preprocessor_obj_path}")

            return transformed_data, self.preprocessor_obj_path

        except Exception as e:
            raise CustomException(e, sys)

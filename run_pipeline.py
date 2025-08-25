import os
import sys
import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import initiate_model_trainer


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


def main():
    try:
        logging.info("Starting the end-to-end model training pipeline.")

        # Define the path to your dataset
        data_path = r"C:\Users\uers\Documents\model_building_excercises\hotel booking customers\data\hotel_booking.csv"
        
        # Data Transformation
        logging.info("Initiating data transformation...")
        data_transformer = DataTransformation()
        transformed_data, preprocessor_path = data_transformer.initiate_data_transformation(data_path)
        logging.info("Data transformation completed successfully.")

        # Model Training
        logging.info("Initiating model training...")
        model_path = initiate_model_trainer(transformed_data)
        logging.info("Model training completed successfully.")

        logging.info(f"Pipeline finished. Final model saved at: {model_path}")
    
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()

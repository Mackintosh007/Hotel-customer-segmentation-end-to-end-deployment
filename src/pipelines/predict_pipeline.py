import os
import sys
import pandas as pd
import pickle

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
    @staticmethod
    def error(message):
        print(f"[ERROR] {message}", file=sys.stderr)
        
def load_object(file_path):
    """
    Loads a Python object from a file using pickle.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


class PredictPipeline:
    
    def __init__(self):
        
        try:
            # Use a path relative to the current file to locate the artifacts directory
            current_dir = os.path.dirname(__file__)
            artifacts_dir = os.path.join(current_dir, '..', 'components', 'artifacts')
            
            preprocessor_path = os.path.join(artifacts_dir, 'preprocessor.pkl')
            model_path = os.path.join(artifacts_dir, 'kmeans_model.pkl')
            
            self.preprocessor = load_object(preprocessor_path)
            self.kmeans_model = load_object(model_path)
            
            logging.info("Preprocessor and K-Means model loaded successfully.")

            # Define the columns to drop, which are used in the predict method
            self.cols_to_drop = [
                'name', 'email', 'phone-number', 'credit_card', 
                'reservation_status', 'reservation_status_date',
                'agent', 'company',
            ]
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features_df):
        
        try:
            logging.info("Starting prediction process.")

            features_df = features_df.drop(columns=self.cols_to_drop)
            features_df = features_df[(features_df['adults'] > 0) | (features_df['children'] > 0) | (features_df['babies'] > 0)]
            features_df['adr'] = features_df['adr'].replace(0, features_df['adr'].mean())
            features_df.dropna(inplace=True)
            features_df['total_nights'] = features_df['stays_in_weekend_nights'] + features_df['stays_in_week_nights']
            features_df = features_df.drop(columns=['stays_in_weekend_nights', 'stays_in_week_nights'])

            logging.info("Applying preprocessor on the new data.")
            transformed_data = self.preprocessor.transform(features_df)

            logging.info("Predicting clusters.")
            clusters = self.kmeans_model.predict(transformed_data)

            features_df['cluster'] = clusters
            logging.info("Prediction complete.")
            return features_df
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    
    input_data_path = r"C:\Users\uers\Documents\model_building_excercises\hotel booking customers\data\hotel_booking.csv"
    
    try:
        logging.info("Command-line execution started.")
        
        # Load the raw data from a file
        raw_df = pd.read_csv(input_data_path)
        
        # Instantiate the pipeline and perform prediction
        predictor = PredictPipeline()
        segmented_df = predictor.predict(raw_df)

        # Save the final segmented DataFrame to a CSV file
        segmented_data_path = os.path.join(os.path.dirname(__file__), '..', 'components', 'artifacts', 'segmented_customers.csv')
        segmented_df.to_csv(segmented_data_path, index=False)
        
        logging.info(f"Prediction pipeline executed successfully. Segmented data saved at: {segmented_data_path}")
    except CustomException as e:
        logging.error(f"An error occurred: {e}")

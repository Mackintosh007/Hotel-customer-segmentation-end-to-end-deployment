import os
import sys
import pandas as pd
import numpy as np
import pickle

def main():
    
    
    print("--- Script execution has started. ---")
    
    # A simple function to load a pickled object from a file
    def load_object(file_path):
        """
        Loads a Python object from a file using pickle.
        """
        try:
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)
        except FileNotFoundError:
            print(f"ERROR: File not found at {file_path}. Please make sure this file is in the same folder as the script.")
            return None
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading {file_path}: {e}")
            return None

    try:
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        DATA_PATH = os.path.join(script_dir, "hotel_booking.csv")
        PREPROCESSOR_PATH = os.path.join(script_dir, "preprocessor.pkl")
        MODEL_PATH = os.path.join(script_dir, "kmeans_model.pkl")

        print("Starting cluster interpretation pipeline...")
        print(f"Attempting to load data from: {DATA_PATH}")

        print("Loading raw data, preprocessor, and trained model...")
        df = pd.read_csv(DATA_PATH)
        preprocessor = load_object(PREPROCESSOR_PATH)
        kmeans_model = load_object(MODEL_PATH)
        
        if df is None or preprocessor is None or kmeans_model is None:
            print("Failed to load one or more necessary files. Exiting.")
            sys.exit(1)
            
        print("Files loaded successfully.")

        print("Applying data cleaning and feature engineering...")
        
        cols_to_drop = [
            'name', 'email', 'phone-number', 'credit_card', 
            'reservation_status', 'reservation_status_date',
            'agent', 'company'
        ]
        df = df.drop(columns=cols_to_drop)
        
        # Handle rows where guests are zero
        df = df[(df['adults'] > 0) | (df['children'] > 0) | (df['babies'] > 0)]
        
        # Impute missing 'adr' values and drop NaNs
        df['adr'] = df['adr'].replace(0, df['adr'].mean())
        df.dropna(inplace=True)
        
        # Create 'total_nights' feature
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df = df.drop(columns=['stays_in_weekend_nights', 'stays_in_week_nights'])
        
        # --- Step 4: Apply Preprocessor and Predict Clusters ---
        print("Transforming data and predicting cluster labels...")
        
        # Use the loaded preprocessor to transform the data
        transformed_data = preprocessor.transform(df)
        
        # Use the loaded model to predict the clusters
        df['cluster'] = kmeans_model.predict(transformed_data)
        
        print("Cluster labels have been added to the DataFrame.")

        print("\n--- Analyzing Customer Segments ---")
        
        # Get the list of numerical columns for analysis
        numerical_cols_for_analysis = df.select_dtypes(include=np.number).columns.tolist()
        numerical_cols_for_analysis.remove('cluster')

        # Group by cluster and calculate the mean for each numerical feature
        cluster_profiles = df.groupby('cluster')[numerical_cols_for_analysis].mean()
        
        print("\nAverage values for each cluster:")
        print(cluster_profiles)
        
        # You can also analyze categorical features
        print("\nTop categories for each cluster:")
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]
            print(f"\n--- Cluster {cluster_id} ---")
            
            # Find the most frequent value for 'market_segment'
            top_market_segment = cluster_df['market_segment'].mode()[0]
            print(f"Market Segment: {top_market_segment}")
            
            # Find the most frequent value for 'customer_type'
            top_customer_type = cluster_df['customer_type'].mode()[0]
            print(f"Customer Type: {top_customer_type}")
            
            # Find the most frequent value for 'arrival_date_month'
            top_month = cluster_df['arrival_date_month'].mode()[0]
            print(f"Busiest Month: {top_month}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Silhouette score import is no longer needed
# from sklearn.metrics import silhouette_score

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


def initiate_model_trainer(transformed_data):
    """
    Trains a K-Means clustering model on the transformed data and saves the model.
    This function also generates and saves the Elbow plot.
    
    Args:
        transformed_data (np.ndarray or pd.DataFrame): The preprocessed data
                                                      ready for clustering.
                                                      
    Returns:
        str: The file path where the trained model is saved.
    """
    try:
        logging.info("Starting model training process.")
        
        # Determine the optimal number of clusters using Elbow Method
        logging.info("Determining optimal clusters with Elbow Method.")
        wcss = []  # Within-Cluster Sum of Squares
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init='auto')
            kmeans.fit(transformed_data)
            wcss.append(kmeans.inertia_)
        
        # Plot the Elbow Method results
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), wcss, marker='o')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.grid(True)
        elbow_path = os.path.join('artifacts', 'elbow_plot.png')
        plt.savefig(elbow_path)
        plt.close()
        logging.info(f"Elbow plot saved at: {elbow_path}")
        
        # Assuming optimal k is found from the plots (e.g., k=4)
        optimal_k = 4
        logging.info(f"Selected optimal number of clusters: {optimal_k}")
        
        # Train the final K-Means model with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init='auto')
        kmeans.fit(transformed_data)
        
        # Save the trained model
        model_path = os.path.join('artifacts', 'kmeans_model.pkl')
        save_object(model_path, kmeans)
        
        logging.info(f"K-Means model saved at: {model_path}")
        
        return model_path

    except Exception as e:
        raise CustomException(e, sys)

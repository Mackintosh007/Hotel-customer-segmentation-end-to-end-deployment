import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

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
    
    try:
        logging.info("Starting model training process.")
        
        # Determine the optimal number of clusters using Elbow Method
        logging.info("Determining optimal clusters with Elbow and Silhouette methods.")
        wcss = []  
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
        elbow_path = os.path.join('src', 'components', 'artifacts', 'elbow_plot.png')
        plt.savefig(elbow_path)
        plt.close()
        logging.info(f"Elbow plot saved at: {elbow_path}")
        
        # Determine the optimal number of clusters using Silhouette Score
        silhouette_scores = []
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init='auto')
            kmeans.fit(transformed_data)
            score = silhouette_score(transformed_data, kmeans.labels_)
            silhouette_scores.append(score)
            
        # Plot the Silhouette Score results
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.title('Silhouette Score For Optimal k')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        silhouette_path = os.path.join('src', 'components', 'artifacts', 'silhouette_plot.png')
        plt.savefig(silhouette_path)
        plt.close()
        logging.info(f"Silhouette plot saved at: {silhouette_path}")
        
        optimal_k = 4
        logging.info(f"Selected optimal number of clusters: {optimal_k}")
        
        # Train the final K-Means model with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init='auto')
        kmeans.fit(transformed_data)
        
        # Save the trained model
        model_path = os.path.join('src', 'components', 'artifacts', 'kmeans_model.pkl')
        save_object(model_path, kmeans)
        
        logging.info(f"K-Means model saved at: {model_path}")
        
        return model_path

    except Exception as e:
        raise CustomException(e, sys)

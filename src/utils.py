import os
import sys
import numpy as np
import pandas as pd
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    
    #Saves a Python object to a file using pickle.
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    
    #Loads a Python object from a file using pickle.
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_kmeans_clusters(x_processed, k_range, evaluation_k=None):
    
    try:
        # Dictionary to store the sum of squared distances for the Elbow Method
        sum_of_sqr_dist = {}
        
        # A variable to hold the model for the final silhouette score calculation
        last_model = None

        for k in k_range:
            logging.info(f"Running K-Means for k = {k}...")
            # Use 'k-means++' for smart initialization to avoid poor convergence
            km = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=42, n_init=10)
            km.fit(x_processed)
            sum_of_sqr_dist[k] = km.inertia_
            logging.info(f"Finished k = {k}. Inertia: {km.inertia_}")
            last_model = km

        # Determine which k to use for the Silhouette Score
        if evaluation_k and evaluation_k in k_range:
            # Re-fit the model for the specific evaluation_k
            km_eval = KMeans(n_clusters=evaluation_k, init='k-means++', max_iter=1000, random_state=42, n_init=10)
            km_eval.fit(x_processed)
            cluster_labels = km_eval.labels_
            logging.info(f"Calculating Silhouette Score for k = {evaluation_k}...")
        else:
            # Use the last model from the loop
            if last_model and len(np.unique(last_model.labels_)) > 1:
                cluster_labels = last_model.labels_
                evaluation_k = k_range.stop - 1 # Use the last k from the range
                logging.info(f"Calculating Silhouette Score for the last evaluated k = {evaluation_k}...")
            else:
                logging.warning("Cannot calculate Silhouette Score: Need at least 2 clusters.")
                silhouette_score_result = None
                return {
                    "inertia_scores": sum_of_sqr_dist,
                    "silhouette_score": silhouette_score_result
                }

        # Calculate the Silhouette Score
        silhouette_score_result = silhouette_score(x_processed, cluster_labels)
        logging.info(f"The Silhouette Score for k={evaluation_k} is: {silhouette_score_result:.4f}")

        return {
            "inertia_scores": sum_of_sqr_dist,
            "silhouette_score": silhouette_score_result
        }

    except Exception as e:
        raise CustomException(e, sys)

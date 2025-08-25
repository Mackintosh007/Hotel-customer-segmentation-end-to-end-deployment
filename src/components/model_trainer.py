import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Import necessary modules from scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs


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

def load_object(file_path):
    """
    Loads a Python object from a file using pickle.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_kmeans_clusters(x_processed, k_range):
    
    try:
        logging.info("Starting K-Means evaluation for optimal k.")
        
        # Dictionary to store evaluation metrics
        evaluation_scores = {
            "inertia_scores": {},
            "silhouette_scores": {}
        }
        
        # Variable to track the best k and its score
        best_k = None
        max_silhouette_score = -1
        
        for k in k_range:
            logging.info(f"Running K-Means for k = {k}...")
            # Skip k=1 as silhouette score is not defined for a single cluster
            if k == 1:
                evaluation_scores["inertia_scores"][k] = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit(x_processed).inertia_
                continue

            km = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=42, n_init=10)
            km.fit(x_processed)
            
            
            evaluation_scores["inertia_scores"][k] = km.inertia_
            
            # Calculate the Silhouette Score
            cluster_labels = km.labels_
            silhouette_avg = silhouette_score(x_processed, cluster_labels)
            evaluation_scores["silhouette_scores"][k] = silhouette_avg
            
            logging.info(f"Finished k = {k}. Inertia: {km.inertia_:.2f}, Silhouette Score: {silhouette_avg:.4f}")
            
            # Check if this k is the best so far
            if silhouette_avg > max_silhouette_score:
                max_silhouette_score = silhouette_avg
                best_k = k

        logging.info(f"Evaluation complete. Optimal k selected based on max Silhouette Score is: {best_k} with a score of {max_silhouette_score:.4f}")
        
        # Add the best k to the report
        evaluation_scores["optimal_k"] = best_k
        return evaluation_scores

    except Exception as e:
        raise CustomException(e, sys)


def initiate_model_trainer():
    
    try:
        logging.info("Starting the K-Means clustering model training pipeline.")
        
        # --- Step 1: Data Ingestion and Preprocessing ---
        logging.info("Generating sample data to demonstrate the pipeline.")
        x, _ = make_blobs(n_samples=1000, centers=5, n_features=10, random_state=42)
        x_df = pd.DataFrame(x)

        logging.info("Scaling the data using StandardScaler.")
        scaler = StandardScaler()
        x_processed = scaler.fit_transform(x_df)
        
        scaler_save_path = os.path.join('artifacts', 'scaler.pkl')
        save_object(scaler_save_path, scaler)
        
        # --- Step 2: Evaluating Models for Optimal 'k' ---
        k_range = range(2, 15)
        evaluation_report = evaluate_kmeans_clusters(x_processed, k_range)
        
        # Extract the optimal k from the report
        optimal_k = evaluation_report["optimal_k"]
        logging.info(f"Selected optimal number of clusters (k) as: {optimal_k}")

        # --- Step 3: Plotting the Results for Visual Analysis ---
        inertia_scores = evaluation_report["inertia_scores"]
        silhouette_scores = evaluation_report["silhouette_scores"]
        
        # Plotting the Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(list(inertia_scores.keys()), list(inertia_scores.values()), marker='o', linestyle='--')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Sum of Squared Distances)')
        plt.xticks(list(inertia_scores.keys()))
        plt.grid(True)
        elbow_plot_path = os.path.join('artifacts', 'elbow_plot.png')
        plt.savefig(elbow_plot_path)
        logging.info(f"Elbow plot saved to {elbow_plot_path}")
        plt.close()
        
        # Plotting the Silhouette Scores
        plt.figure(figsize=(10, 6))
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
        plt.title('Silhouette Score for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.xticks(list(silhouette_scores.keys()))
        plt.grid(True)
        silhouette_plot_path = os.path.join('artifacts', 'silhouette_plot.png')
        plt.savefig(silhouette_plot_path)
        logging.info(f"Silhouette score plot saved to {silhouette_plot_path}")
        plt.close()
        
        # --- Step 4: Train the Final K-Means Model ---
        logging.info(f"Training the final K-Means model with k = {optimal_k}")
        final_model = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=1000, random_state=42, n_init=10)
        final_model.fit(x_processed)
        
        # --- Step 5: Save the Final Model ---
        model_save_path = os.path.join('artifacts', 'kmeans_model.pkl')
        save_object(model_save_path, final_model)
        
        logging.info(f"Final K-Means model saved to {model_save_path}")
        
        # Return the path to the saved model for further use
        return model_save_path
    
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    initiate_model_trainer()

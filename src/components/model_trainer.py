import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import necessary modules from scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Import custom modules from your project structure
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_kmeans_clusters, save_object

def initiate_model_trainer():
    """
    This function orchestrates the entire K-Means clustering workflow:
    1. Loads and preprocesses the data.
    2. Evaluates the best number of clusters (k) using the Elbow Method.
    3. Trains the final K-Means model with the optimal k.
    4. Saves the trained model to a pickle file.
    """
    try:
        logging.info("Starting the K-Means clustering model training pipeline.")
        
        # --- Step 1: Data Ingestion and Preprocessing ---
        # NOTE: For this demonstration, we'll generate sample data.
        # In a real project, you would load your data here.
        logging.info("Generating sample data to demonstrate the pipeline.")
        x, _ = make_blobs(n_samples=1000, centers=5, n_features=10, random_state=42)
        x_df = pd.DataFrame(x)

        logging.info("Scaling the data using StandardScaler.")
        scaler = StandardScaler()
        x_processed = scaler.fit_transform(x_df)
        
        # Save the scaler object as it's needed for future data processing
        scaler_save_path = os.path.join('artifacts', 'scaler.pkl')
        save_object(scaler_save_path, scaler)
        logging.info(f"Scaler object saved to {scaler_save_path}")
        
        # --- Step 2: Evaluate Models for Optimal 'k' ---
        # We'll evaluate k from 1 to 10
        k_range = range(1, 11)
        logging.info(f"Evaluating K-Means models for k in range {list(k_range)}")
        
        evaluation_report = evaluate_kmeans_clusters(x_processed, k_range)
        
        # --- Step 3: Find the Optimal 'k' from the Report ---
        inertia_scores = evaluation_report["inertia_scores"]
        
        # Plotting the Elbow Method to visually find the optimal k
        plt.figure(figsize=(10, 6))
        plt.plot(list(inertia_scores.keys()), list(inertia_scores.values()), marker='o', linestyle='--')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Sum of Squared Distances)')
        plt.xticks(list(inertia_scores.keys()))
        plt.grid(True)
        # Save the plot for later analysis
        elbow_plot_path = os.path.join('artifacts', 'elbow_plot.png')
        plt.savefig(elbow_plot_path)
        logging.info(f"Elbow plot saved to {elbow_plot_path}")

        # Based on the plot, we manually choose the optimal k.
        # For our generated data, the elbow is clearly at k=5.
        optimal_k = 5
        logging.info(f"Selected optimal number of clusters (k) as: {optimal_k}")

        # Optionally, get the Silhouette Score for this optimal_k
        silhouette_score_result = evaluate_kmeans_clusters(
            x_processed, 
            k_range=range(optimal_k, optimal_k + 1), 
            evaluation_k=optimal_k
        )["silhouette_score"]
        
        logging.info(f"Silhouette Score for the optimal k={optimal_k} is: {silhouette_score_result:.4f}")

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

# Main execution block
if __name__ == "__main__":
    initiate_model_trainer()

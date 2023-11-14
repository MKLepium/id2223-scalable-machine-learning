import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def generate_samples_with_quality(csv_file, num_samples):
    df = pd.read_csv(csv_file, delimiter=';')
    columns = df.columns
    samples = {}
    
    for column in columns:
        if column == 'quality':
            # Set arbitrary quality values to 0 for all samples
            samples[column] = np.zeros(num_samples)
        else:
            # Calculate the mean and standard deviation for each column (excluding 'quality')
            mean = df[column].mean()
            std = df[column].std()
            
            # Generate random samples based on a normal distribution with the calculated mean and std
            samples[column] = np.random.normal(mean, std, num_samples)
    
    generated_df = pd.DataFrame(samples)
    
    # Exclude the 'quality' column for K-Means clustering
    kmeans_columns = [col for col in columns if col != 'quality']
    kmeans_data = generated_df[kmeans_columns]
    
    # Fit a K-Means model to the data without the 'quality' column
    kmeans = KMeans(n_clusters=5, random_state=0).fit(kmeans_data)
    
    # Predict cluster labels for the generated data
    generated_cluster_labels = kmeans.predict(kmeans_data)
    
    # Calculate the cluster centers for the generated data
    cluster_centers = kmeans.cluster_centers_
    
    # Update the 'quality' column in the generated DataFrame based on the cluster centers
    generated_df['quality'] = cluster_centers[generated_cluster_labels]
    
    return generated_df

white_wine_path = 'wine+quality/winequality-white.csv'
red_wine_path = 'wine+quality/winequality-red.csv'

# Generate 100 samples for white wine data with arbitrary quality values
white_wine_samples = generate_samples_with_quality(white_wine_path, num_samples=100)
# Generate 100 samples for red wine data with arbitrary quality values
red_wine_samples = generate_samples_with_quality(red_wine_path, num_samples=100)

print("White wine generated samples with adjusted quality: \n", white_wine_samples)
print()
print("Red wine generated samples with adjusted quality: \n", red_wine_samples)

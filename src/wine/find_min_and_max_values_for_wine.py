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
            
            # Generate random samples based on a log-normal distribution with the calculated mean and std
            samples[column] = np.random.normal(mean, std, num_samples)
            # Replace negative values with 0
            samples[column][samples[column] < 0] = 0
            
    
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

def generate_n_wine_samples(num_samples):
    """
    Generates n/2 samples of wine using the k-means clustering algorithm to generate the quality values
    And a normal distribution to generate the other values
    """
    white_wine_path = 'wine+quality/winequality-white.csv'
    red_wine_path = 'wine+quality/winequality-red.csv'

    
    white_wine_samples = generate_samples_with_quality(white_wine_path, num_samples=int(num_samples/2))
    white_wine_samples['type'] = -1
    red_wine_samples = generate_samples_with_quality(red_wine_path, num_samples=int(num_samples/2))
    red_wine_samples['type'] = 1

    #print("White wine generated samples with adjusted quality: \n", white_wine_samples)
    #print()
    #print("Red wine generated samples with adjusted quality: \n", red_wine_samples)

    # Replace spaces with underscores in the column names
    white_wine_samples.columns = [col.replace(' ', '_') for col in white_wine_samples.columns]
    red_wine_samples.columns = [col.replace(' ', '_') for col in red_wine_samples.columns]




    return pd.concat([white_wine_samples, red_wine_samples])


if __name__ == "__main__":
    generate_n_wine_samples(200)
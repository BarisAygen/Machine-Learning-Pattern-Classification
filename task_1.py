import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Load the data
dataset = dict()

with open("feature_names.txt") as f:
    content = f.readlines()

for i in range(len(content)):
    content[i] = content[i][:-1]

for feature_name in content:
    dataset[feature_name] = []
dataset['label'] = []

# initialize an empty array
avg_intra_class_variations = []

directory = ['comcuc/', 'cowpig1/', 'eucdov/', 'eueowl1/', 'grswoo/', 'tawowl1/']

for dir_path in directory:
    file_labels = [f for f in os.listdir(dir_path) if f.endswith('.labels.npy')]
    file_features = [f for f in os.listdir(dir_path) if not f.endswith('.labels.npy')]

    for i in range(len(file_labels)):
        path_features = os.path.join(dir_path, file_features[i])
        path_labels = os.path.join(dir_path, file_labels[i])

        features = np.load(path_features)
        labels = np.load(path_labels)

        label_column = list(labels[:, 0])
        dataset['label'] += label_column

        for k in range(features.shape[1]):
            values = list(features[:, k])
            dataset[content[k]] += values

    df_1 = pd.DataFrame(dataset)
    X_1 = df_1.iloc[:, :-1]
    y_1 = df_1.iloc[:, -1]

    # Normalize the features using StandardScaler
    scaler_1 = StandardScaler()
    X_1 = scaler_1.fit_transform(X_1)

    # Intra-class variation
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)
    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    model.fit(X_train_1)

    # Compute the intra-class variation for each class
    intra_class_variation = []
    for i in range(2):
        class_data = X_train_1[model.labels_ == i]
        intra_class_variation.append(np.mean(np.var(class_data, axis=0)))

    # Compute the average intra-class variation across all classes
    avg_intra_class_variation = np.mean(intra_class_variation)

    print("Intra-class variation:", intra_class_variation)
    print("Average intra-class variation:", avg_intra_class_variation)

    # append the average intra-class variation to the array
    avg_intra_class_variations.append(avg_intra_class_variation)

# plot a histogram of the array
classes = ['comcuc', 'cowpig1', 'eucdov', 'eueowl1', 'grswoo', 'tawowl1']
plt.bar(classes, avg_intra_class_variations)
plt.title('Intra-class Variation')
plt.xlabel('Classes')
plt.ylabel('avg_intra_class_variations')
plt.show()

df = pd.DataFrame(dataset)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Visiulize the data
print(df)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Inter-class variation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KMeans(n_clusters=7, random_state=42, n_init=10)
model.fit(X_train)

# Compute the centroids of each class
centroids = model.cluster_centers_

# Compute the distance between the centroids of each class
inter_class_distance = 0
for i in range(centroids.shape[0]):
    for j in range(i + 1, centroids.shape[0]):
        inter_class_distance += np.linalg.norm(centroids[i] - centroids[j])

# Normalize the inter-class distance by the number of class combinations
n_combinations = (centroids.shape[0] * (centroids.shape[0] - 1)) / 2
inter_class_distance /= n_combinations

print("Inter-class distance:", inter_class_distance)

# X is the feature matrix
corr_matrix = np.corrcoef(X.T)

# Set the threshold for highly correlated pairs
corr_threshold = 0.95

# Find highly correlated pairs of features
corr_pairs = []
for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[1]):
        if abs(corr_matrix[i, j]) > corr_threshold:
            corr_pairs.append((i, j, abs(corr_matrix[i, j])))

# Sort the pairs by correlation coefficient in descending order
corr_pairs.sort(key=lambda x: x[2], reverse=True)

# Select the top 5 pairs
best_corr_pairs = corr_pairs[:5]

# Print the names of the selected features
print("Selected Pairs:")
for pair in best_corr_pairs:
    feature1_name = content[pair[0]]
    feature2_name = content[pair[1]]
    print("-", feature1_name, "and", feature2_name, ":", pair[2])

# X is the feature matrix and y is the label vector
k = 5

# Calculate absolute correlation coefficients between each feature and the label
corr_coefs = [abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])]

# Get indices of the top k features with highest absolute correlation coefficients
top_k_indices = sorted(range(len(corr_coefs)), key=lambda i: corr_coefs[i], reverse=True)[:k]

# Print the names of the selected features
print("Selected Features:")
for i in range(k):
    feature_idx = top_k_indices[i]
    feature_name = content[feature_idx]
    feature_value = X[0][feature_idx]
    print("-", feature_name, ":", feature_value)

# Select the top 5 features with the highest mutual information score
selector = SelectKBest(mutual_info_classif, k=5)
X_new = selector.fit_transform(X, y)

# Get the names of the selected features
mask = selector.get_support()  # list of booleans
selected_features_indices = mask.nonzero()[0]  # indices of selected features
selected_features = X[:, selected_features_indices]  # selected features

# Print the names of the selected features
print("Selected Features:")
for i, feature_idx in enumerate(selected_features_indices):
    feature_name = content[feature_idx]
    feature_value = selected_features[0][i]
    print("-", feature_name, ":", feature_value)
    if i == 4:
        break
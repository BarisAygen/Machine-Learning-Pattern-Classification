import os
import joblib
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot, pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import NearMiss
from collections import Counter

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

df = pd.DataFrame(dataset)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Display data
print('Original data:')
display(df)

# Normalize the features using StandardScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
df.iloc[:, :-1] = X

# Display data after normalization
print('Data after normalization:')
display(df)

# X is the feature matrix
corr_matrix = np.corrcoef(X.T)

# Set the threshold for highly correlated pairs
corr_threshold = 0.8

# Find highly correlated pairs of features to take one of them out
corr_pairs = []
for i in range(corr_matrix.shape[0]):
    for j in range(i + 1, corr_matrix.shape[1]):
        if abs(corr_matrix[i, j]) > corr_threshold:
            corr_pairs.append((i, j, abs(corr_matrix[i, j])))

# Sort the pairs by correlation coefficient in descending order
corr_pairs.sort(key=lambda x: x[2], reverse=True)

# Select the top 10 pairs
best_corr_pairs = corr_pairs[:10]

# Print the names of the selected features
print("Selected pairs according to correlation coefficient:")
for pair in best_corr_pairs:
    feature1_name = content[pair[0]]
    feature2_name = content[pair[1]]
    print("-", feature1_name, "and", feature2_name, ":", pair[2])

#     # From data remove one of the correlated pair
#     df = df.drop(feature2_name, axis=1)
#
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]

# Calculate absolute correlation coefficients between each feature and the label with pearson
corr_coefs = [abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])]

# Get indices of the top k features with highest absolute correlation coefficients
top_k_indices = sorted(range(len(corr_coefs)), key=lambda i: corr_coefs[i], reverse=True)[:10]

# Print the names of the selected features
print("Selected features according to the pearson score:")
for i in range(10):
    feature_idx = top_k_indices[i]
    feature_name = content[feature_idx]
    feature_value = X[0][feature_idx]
    print("-", feature_name, ":", feature_value)

# Get the names of the selected features
selected_features = [content[idx] for idx in top_k_indices[:10]]

# Get the correlation scores
correlation_scores = [corr_coefs[idx] for idx in top_k_indices[:10]]

# Plot the correlation scores
plt.figure(figsize=(10, 6))
plt.bar(selected_features, correlation_scores)
plt.xlabel('Features')
plt.ylabel('Pearson Correlation Score')
plt.title('Top 10 Features Selected by Pearson Correlation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Select the top 10 features with the highest mutual information score
selector = SelectKBest(mutual_info_classif, k=10)
X_new = selector.fit_transform(X, y)

# Get the names of the selected features
mask = selector.get_support()  # list of booleans
selected_features_indices = mask.nonzero()[0]  # indices of selected features
selected_features = X[:, selected_features_indices]  # selected features

# Print the names of the selected features
print("Selected features according to the mutual information score:")
for i, feature_idx in enumerate(selected_features_indices):
    feature_name = content[feature_idx]
    feature_value = selected_features[0][i]
    print("-", feature_name, ":", feature_value)
    if i == 9:
        break

# Get the names of the selected features
selected_features = [content[idx] for idx in selected_features_indices[:10]]

# Get the mutual information scores
mutual_info_scores = selector.scores_[selected_features_indices[:10]]

# Plot the mutual information scores
plt.figure(figsize=(10, 6))
plt.bar(selected_features, mutual_info_scores)
plt.xlabel('Features')
plt.ylabel('Mutual Information Score')
plt.title('Top 10 Features Selected by Mutual Information')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

best_features = []

# Add the selected features from Pearson correlation to the best_features list
for i in top_k_indices:
    feature_name = content[i]
    if feature_name not in best_features:
        best_features.append(feature_name)

# Add the selected features from mutual information to the best_features list
for i in selected_features_indices:
    feature_name = content[i]
    if feature_name not in best_features:
        best_features.append(feature_name)

# best_features=['cln_flatness_mean', 'cln_contrast_mean_4', 'cln_centroid_mean','cln_contrast_mean_3', 'raw_energy_std', 'raw_mfcc_std_1',
#        'cln_mfcc_mean_0', 'raw_contrast_mean_3', 'cln_mfcc_std_1','raw_mfcc_std_0', 'yin_4', 'yin_5', 'yin_6', 'yin_7', 'yin_8', 'yin_9',
#        'yin_10', 'yin_11']

# Initialize new dataset with only selected columns (features)
dict_features = dict()

for i in range(len(best_features)):
    for row in dataset:
        if row == best_features[i]:
            dict_features[best_features[i]] = dataset[row]

dict_features['label'] = dataset['label']

df_new = pd.DataFrame(dict_features)

# Display data with selected features
print('Data with selected features:')
display(df_new)

# Find duplicate columns
duplicated_cols = df_new.columns[df_new.T.duplicated()]

# Drop duplicate columns
df_new = df_new.drop(duplicated_cols, axis=1)

X1 = df_new.iloc[:, :-1]
y2 = df_new.iloc[:, -1]

# Summarize class distribution
print('Class distribution:')
counter = Counter(y)
print(counter)

# Scatter plot of examples by class label
class_counts = df_new['Label'].value_counts()
class_labels = class_counts.index

plt.bar(class_labels, class_counts)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# Define the undersampling method
undersample = NearMiss(version=1, n_neighbors=3)

# Transform the dataset
X1, y2 = undersample.fit_resample(X1, y2)
# Create a new DataFrame with selected features and undersampled data
df_new = pd.DataFrame(X1, columns=best_features)
df_new['label'] = y2

# summarize the new class distribution after undersampling
print('Class distribution after undersampling:')
counter = Counter(y)
print(counter)

# Scatter plot of examples by class label after undersampling
# Scatter plot of examples by class label
class_counts = df_new['label'].value_counts()
class_labels = class_counts.index

plt.bar(class_labels, class_counts)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution after undersampling')
plt.show()

# Separate the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X1, y2, test_size=0.2, random_state=42)

# # Assign higher weight to minority class
# class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

# Create the KFold object
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Create a decision tree object
tree = DecisionTreeClassifier(random_state=42)

# Set the parameter grid
param_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
              'max_depth': [8, 12, 16],
              'max_features': ['sqrt', 'log2']}

# Create the grid search object
random_search = RandomizedSearchCV(tree, param_grid, cv=kf, n_jobs=-1)

# Fit the grid search object to the training data
random_search.fit(X_train, y_train)

# Print the best parameter combination and the corresponding accuracy score
print("Decision tree:")
print("Best parameters: ", random_search.best_params_)
print("Best cross-validation score: {:.2f}".format(random_search.best_score_))
print("Test set score: {:.2f}".format(random_search.score(X_test, y_test)))

# Create the classifier with specified parameters
classifier = DecisionTreeClassifier(criterion='gini', max_depth=12, max_features='sqrt', random_state=42,
                                    # class_weight=class_weights
                                    )

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the accuracy of the classifier on the test set
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: ', cm)
print('Accuracy Score: ', accuracy_score(y_test, y_pred))
print('F1 Score: ', f1_score(y_test, y_pred, average=None))

# Save the trained model to a file
joblib.dump(classifier, 'model.pkl')

# Load saved classifier
model = joblib.load('model.pkl')

# List of test file names
test_files = ['test00.npy', 'test01.npy', 'test02.npy', 'test03.npy', 'test04.npy', 'test05.npy', 'test06.npy',
              'test07.npy',
              'test08.npy', 'test09.npy', 'test10.npy', 'test11.npy', 'test12.npy', 'test13.npy', 'test14.npy',
              'test15.npy']

# Create an empty list to store the test arrays
test_arrays = []

# Load and store the test arrays
for file in test_files:
    file_path = os.path.join('Tests', file)
    test_data = np.load(file_path)
    test_arrays.append(test_data)

# Concatenate the test arrays into a single array
combined_array = np.concatenate(test_arrays)

# Create a DataFrame from the combined array
df_test = pd.DataFrame(combined_array)

# Display test data
print('Test data:')
display(df_test)

# Normalize the tests
df_test = scaler.fit_transform(df_test)

# Display test data after normalization
print('Test data after normalization:')
display(df_test)

tmp = dict()
for i in range(len(content)):
    tmp[content[i]] = df_test[:, i]

test_values = dict()

for i in range(len(best_features)):
    if best_features[i] not in test_values:
        test_values[best_features[i]] = []
        test_values[best_features[i]] += tmp[best_features[i]].tolist()
    else:
        test_values[best_features[i]] += tmp[best_features[i]].tolist()

test_values = pd.DataFrame(test_values)

# Display test data when it is ready for prediction
print('Final version of test data before estimation phase:')
display(test_values)

# Make predictions on the test data
predictions = model.predict(test_values)

# Save results on excel file
csv_file_path = 'tree_file.csv'

df_predictions = pd.DataFrame()

for prediction in predictions:
    df_prediction = pd.DataFrame(prediction.reshape(1, -1))

    df_predictions = pd.concat([df_predictions, df_prediction], ignore_index=True)

df_predictions.to_csv(csv_file_path, index=False)

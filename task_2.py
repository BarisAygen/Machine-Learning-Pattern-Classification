import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Initialize the dataset
dataset = dict()

with open("feature_names.txt") as f:
    content = f.readlines()

for i in range(len(content)):
    content[i] = content[i][:-1]

for feature_name in content:
    dataset[feature_name] = []
dataset['label'] = []

directory = [
    'comcuc/',
    'cowpig1/',
    'eucdov/',
    'eueowl1/'
    'grswoo/',
    'tawowl1/']

# For each bird
for dir_path in directory:
    file_labels = [f for f in os.listdir(dir_path) if f.endswith('.labels.npy')]
    file_features = [f for f in os.listdir(dir_path) if not f.endswith('.labels.npy')]

    print(dir_path[:-1], ":")

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

    df = pd.DataFrame(dataset)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Visiulize the data
    print(df)

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

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

    # Initialize new dataset with only selected columns (features)
    dict_features = dict()

    for i in range(len(best_features)):
        for row in dataset:
            if row == best_features[i]:
                dict_features[best_features[i]] = dataset[row]

    dict_features['label'] = dataset['label']

    df_new = pd.DataFrame(dict_features)
    X1 = df_new.iloc[:, :-1]
    y2 = df_new.iloc[:, -1]

    # Find duplicate columns
    duplicated_cols = df_new.columns[df_new.T.duplicated()]

    # Drop duplicate columns
    df_new = df_new.drop(duplicated_cols, axis=1)

    # Visualize the data
    print(df_new)

    # Visualize the data
    print(df_new.columns)

    # Separate the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X1, y2, test_size=0.2, random_state=42)

    # Create the KFold object
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Create the SVM classifier
    svm_classifier = SVC(random_state=42)

    # Set the parameter grid
    param_grid1 = {'C': [30, 40, 50],
                  'gamma': [0.02, 0.1, 0.2],
                  'kernel': ['rbf', 'linear']}

    # Create the grid search object
    random_search = RandomizedSearchCV(svm_classifier, param_distributions=param_grid1, cv=kf, scoring='F1',
                                       n_jobs=-1)

    # Fit the grid search object to the training data
    random_search.fit(X_train, y_train)

    # Print the best parameter combination and the corresponding accuracy score
    print("SVM:")
    print("Best parameters: ", random_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(random_search.best_score_))
    print("Test set score: {:.2f}".format(random_search.score(X_test, y_test)))

    # Create a decision tree object
    tree = DecisionTreeClassifier(random_state=42)

    # Set the parameter grid
    param_grid2 = {'criterion': ['gini', 'entropy', 'log_loss'],
                   'max_depth': [4, 8, 12],
                   'max_features': ['sqrt', 'log2']}

    # Create the grid search object
    random_search2 = RandomizedSearchCV(tree, param_grid2, cv=kf, scoring='accuracy', n_jobs=-1)

    # Fit the grid search object to the training data
    random_search2.fit(X_train, y_train)

    # Print the best parameter combination and the corresponding accuracy score
    print("Decision tree:")
    print("Best parameters: ", random_search2.best_params_)
    print("Best cross-validation score: {:.2f}".format(random_search2.best_score_))
    print("Test set score: {:.2f}".format(random_search2.score(X_test, y_test)))

    # Create the KNN classifier
    knn_classifier = KNeighborsClassifier()

    # Set the parameter grid
    param_grid3 = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    # Create the grid search object
    random_search3 = RandomizedSearchCV(knn_classifier, param_distributions=param_grid3, cv=kf, scoring='accuracy',
                                       n_jobs=-1)

    # Fit the grid search object to the training data
    random_search3.fit(X_train, y_train)

    # Print the best parameter combination and the corresponding accuracy score
    print("KNN:")
    print("Best parameters: ", random_search3.best_params_)
    print("Best cross-validation score: {:.2f}".format(random_search3.best_score_))
    print("Test set score: {:.2f}".format(random_search3.score(X_test, y_test)))

    # Create the Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Set the parameter grid
    param_grid4 = {
        'n_estimators': [75, 100, 125],
        'max_depth': [4, 8, 12],
        'max_features': ['log2', 'sqrt']
    }

    # Create the grid search object
    random_search4 = RandomizedSearchCV(rf_classifier, param_distributions=param_grid4, cv=kf, scoring='accuracy',
                                       n_jobs=-1)

    # Fit the grid search object to the training data
    random_search4.fit(X_train, y_train)

    # Print the best parameter combination and the corresponding accuracy score
    print("Random Forest:")
    print("Best parameters: ", random_search4.best_params_)
    print("Best cross-validation score: {:.2f}".format(random_search4.best_score_))
    print("Test set score: {:.2f}".format(random_search4.score(X_test, y_test)))

    # Create the Linear Discriminant Analysis classifier
    lda_classifier = LinearDiscriminantAnalysis()

    # Set the parameter grid
    param_grid5 = {
        'solver': ['lsqr', 'eigen'],
        'shrinkage': ['auto', None, 0.5],
        'store_covariance': [True, False]
    }

    # Create the grid search object
    random_search5 = RandomizedSearchCV(lda_classifier, param_distributions=param_grid5, cv=kf, scoring='accuracy',
                                       n_jobs=-1)

    # Fit the grid search object to the training data
    random_search5.fit(X_train, y_train)

    # Print the best parameter combination and the corresponding accuracy score
    print("Linear Discriminant Analysis:")
    print("Best parameters: ", random_search5.best_params_)
    print("Best cross-validation score: {:.2f}".format(random_search5.best_score_))
    print("Test set score: {:.2f}".format(random_search5.score(X_test, y_test)))

    # For only one bird
    break
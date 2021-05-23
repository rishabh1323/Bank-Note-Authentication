# Importing libraries
import pylab
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix

warnings.filterwarnings('ignore')

# Reading the dataset
df = pd.read_csv('BankNote_Authentication.csv')
print(df.head())

# Checking for null values
print(df.isnull().sum())

# Getting some stats about data
print(df.describe())

# Extracting dependent and independent features
X = df.drop('class', axis=1)
Y = df['class']

# Plotting histograms
figure, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.suptitle('Histograms of Original Input Features (Without Scaling)')
sns.histplot(data=X['variance'], kde=True, ax=axes[0, 0], bins=50)
sns.histplot(data=X['skewness'], kde=True, ax=axes[0, 1], bins=50)
sns.histplot(data=X['curtosis'], kde=True, ax=axes[1, 0], bins=50)
sns.histplot(data=X['entropy'], kde=True, ax=axes[1, 1], bins=50)
plt.show()

plt.figure(figsize=(15, 10))
sns.boxplot(data=X)
plt.title('Boxplot for all Input Features')
plt.show()

# Plotting the heat map of correlation matrix for input features
plt.figure(figsize=(10, 8))
sns.heatmap(data=X.corr(), annot=True)
plt.title('HeatMap of Correlation Matrix of Independent Features')
plt.show()

# Function to plot histogram and Q-Q plot
def plot_QQ(df, feature):
    plt.figure(figsize=(12, 6))
    plt.suptitle(feature)
    plt.subplot(1, 2, 1)
    plt.title('Histogram')
    sns.histplot(data=df[feature], bins=50)
    plt.subplot(1, 2, 2)
    stats.probplot(df[feature], dist='norm', plot=pylab)
    plt.show()

# Plotting histograms and Q-Q plots
for feature in X.columns:
    plot_QQ(X, feature)

# Splitting the dataset into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=0)
print('Shape of X_train :', X_train.shape)
print('Shape of y_train :', y_train.shape)
print('Shape of X_test  :', X_test.shape)
print('Shape of y_test  :', y_test.shape)

# Applying standard scaling to training input features
standard_scaler = StandardScaler()
X_train = pd.DataFrame(standard_scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(standard_scaler.transform(X_test), columns=X_test.columns)

# Plotting histograms
figure, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.suptitle('Histograms of Input Features after Scaling')
sns.histplot(data=X_train['variance'], kde=True, ax=axes[0, 0], bins=50)
sns.histplot(data=X_train['skewness'], kde=True, ax=axes[0, 1], bins=50)
sns.histplot(data=X_train['curtosis'], kde=True, ax=axes[1, 0], bins=50)
sns.histplot(data=X_train['entropy'], kde=True, ax=axes[1, 1], bins=50)
plt.show()

# Defining hyperparameter values to tune over
grid_params = {
    'C' : 10.0 ** np.arange(-2, 2),
    'penalty' : ['l1', 'l2'],
    'class_weight' : [{0:1, 1:10}, {0:1, 1:100}, {0:1, 1:1000}, {0:1, 1:10000}]
}

# Instantiating and training a Logistic Regression model
logistic_classifier = GridSearchCV(LogisticRegression(n_jobs=-1), param_grid=grid_params, cv=3, n_jobs=-1)
logistic_classifier.fit(X_train, y_train)

# Printing hyperparameter values for best logistic regression model
print('best hyperparamters:', logistic_classifier.best_params_)

# Creating final logistic regression model with best hyperparameter values
logistic_classifier = LogisticRegression(C=10.0, class_weight={0:1,1:10}, penalty='l2', n_jobs=-1)
logistic_classifier.fit(X_train, y_train)

# Predicting on test data and printing metrics
y_pred = logistic_classifier.predict(X_test)
print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Plotting confusion matrix for visualization
plot_confusion_matrix(logistic_classifier, X_test, y_test)
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

# Defining hyperparameter values to tune over
grid_params = {
    'C' : 10.0 ** np.arange(-2, 2),
    'kernel' : ['linear', 'rbf'],
    'class_weight' : [{0:1, 1:10}, {0:1, 1:100}, {0:1, 1:1000}, {0:1, 1:10000}]
}

# Instantiating and training a Logistic Regression model
support_vector_classifier = GridSearchCV(SVC(), param_grid=grid_params, cv=3, n_jobs=-1)
support_vector_classifier.fit(X_train, y_train)

# Printing hyperparameter values for best logistic regression model
print('best hyperparamters:', support_vector_classifier.best_params_)

# Creating final logistic regression model with best hyperparameter values
support_vector_classifier = SVC(C=10.0, class_weight={0:1,1:10}, kernel='rbf')
support_vector_classifier.fit(X_train, y_train)

# Predicting on test data and printing metrics
y_pred = support_vector_classifier.predict(X_test)
print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Plotting confusion matrix for visualization
plot_confusion_matrix(support_vector_classifier, X_test, y_test)
plt.title('Confusion Matrix for Support Vector Classifier')
plt.show()

# Defining hyperparameter values to tune over
grid_params = {
    'n_estimators' : [100, 250, 500],
    'class_weight' : [{0:1, 1:1}, {0:1, 1:100}, {0:1, 1:1000}]
}

# Instantiating and training a Logistic Regression model
random_forest_classifier = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid=grid_params, cv=3, n_jobs=-1)
random_forest_classifier.fit(X_train, y_train)

# Printing hyperparameter values for best logistic regression model
print('best hyperparamters:', random_forest_classifier.best_params_)

# Creating final logistic regression model with best hyperparameter values
random_forest_classifier = RandomForestClassifier(n_estimators=250, class_weight={0:1,1:1}, n_jobs=-1)
random_forest_classifier.fit(X_train, y_train)

# Predicting on test data and printing metrics
y_pred = random_forest_classifier.predict(X_test)
print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Plotting confusion matrix for visualization
plot_confusion_matrix(random_forest_classifier, X_test, y_test)
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()

# Exporting the models and standard scaler to pickle files

with open('models.pkl', 'wb') as file:
    pickle.dump([logistic_classifier, support_vector_classifier, random_forest_classifier], file)
    
with open('standard_scaler.pkl', 'wb') as file:
    pickle.dump(standard_scaler, file)
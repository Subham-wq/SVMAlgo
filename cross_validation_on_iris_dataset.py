# Importing the required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# Load the dummy dataset
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

# Create an SVM classifier
svm_classifier = SVC(kernel='linear')

# Perform cross-validation
scores = cross_val_score(svm_classifier, X, y, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Importing the required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load the dummy dataset
dataset = datasets.load_boston()
X = dataset.data
y = dataset.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM regressor
svm_regressor = SVR(kernel='linear')

# Train the SVM regressor
svm_regressor.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = svm_regressor.predict(X_test)

# Evaluate the performance of the regressor using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

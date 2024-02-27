import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming 'data' is your numpy array where the first 6 columns are features and the 7th column is the target
# Replace this with your actual data
data = np.load("whole_res.npy")

# Splitting the data into features and target
X = data[:, :6]
y = data[:, 6]

# Splitting data into train, dev, and test sets (60%, 20%, 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Training a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(model.coef_)

# Evaluating the model on the development set
y_pred_dev = model.predict(X_dev)
accuracy_dev = accuracy_score(y_dev, y_pred_dev)
print("Development Set Accuracy:", accuracy_dev)

# Finally, evaluate on the test set
# y_pred_test = model.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_pred_test)
# print("Test Set Accuracy:", accuracy_test)
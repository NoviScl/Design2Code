import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your data
data = np.load("whole_res.npy")

# Splitting the data into features and target
X = data[:, 1:6]
y = data[:, 6]

# Add a constant to the model (intercept)
# X = sm.add_constant(X)

# Splitting data into train, dev, and test sets (60%, 20%, 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
# X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and fit the model
model = sm.Logit(y_train, X_train)
result = model.fit()

# Print the summary of the model to see the p-values along with coefficients
print(result.summary())

# Predictions and evaluation (using sklearn's accuracy_score for consistency with your original code)
y_pred_dev = result.predict(X_temp)
# Converting probabilities to class labels (0 or 1) based on a 0.5 threshold
y_pred_dev = (y_pred_dev > 0.5).astype(int)
accuracy_dev = accuracy_score(y_temp, y_pred_dev)
print("Development Set Accuracy:", accuracy_dev)

# Finally, evaluate on the test set
# y_pred_test = model.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_pred_test)
# print("Test Set Accuracy:", accuracy_test)

import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold

# Load your data
data = np.load("whole_res.npy")

# Splitting the data into features and target
X = data[:, 1:6]
y = data[:, 6]

# Add a constant to the model (intercept)
# X = sm.add_constant(X)

# Define the number of folds for cross-validation
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

coefficients = []
p_values = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create and fit the model
    model = sm.Logit(y_train, X_train)
    result = model.fit(disp=0)  # disp=0 suppresses the fitting output

    # Store the coefficients and p-values
    coefficients.append(result.params)
    p_values.append(result.pvalues)

# Convert lists to numpy arrays for easier analysis
coefficients = np.array(coefficients)
p_values = np.array(p_values)

# Calculate average and standard deviation of coefficients
avg_coefficients = np.mean(coefficients, axis=0)
std_coefficients = np.std(coefficients, axis=0)

# Analyze p-values
# You can decide on a threshold for significance, e.g., 0.05
significant_predictors = np.mean(p_values < 0.05, axis=0)

# Print the results
print("Average Coefficients:\n", avg_coefficients)
print("\nStandard Deviation of Coefficients:\n", std_coefficients)
print("\nProportion of folds where each predictor is significant (p < 0.05):\n", significant_predictors)
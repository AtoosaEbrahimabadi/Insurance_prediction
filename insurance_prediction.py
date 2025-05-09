import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "D:/University-Prj/ML0/insurance.csv"
insurance_data = pd.read_csv(file_path)

# Display basic info (optional, but good for debugging)
print(insurance_data.head())
print(insurance_data.info())
print(insurance_data.isnull().sum())

# Convert categorical variables using one-hot encoding
insurance_data = pd.get_dummies(insurance_data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Prepare the data
X = insurance_data.drop('charges', axis=1)
y = insurance_data['charges']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features.  Important for many models, including RF if you use features like polynomial features later.
numerical_features = ['age', 'bmi', 'children']  # List numerical features BEFORE scaling
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])


# Define the parameter grid for GridSearchCV.  Important to tune these for best performance!
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [4, 6, 8, 10],       # Maximum depth of the trees
    'min_samples_split': [2, 4, 6],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 3]     # Minimum number of samples required to be at a leaf node
}


# Instantiate the Random Forest Regressor
rf = RandomForestRegressor(random_state=42) #keep random_state for consistent results

# Instantiate GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1) #n_jobs=-1 to utilize all cores

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_

print(f"Best Parameters: {best_params}")

# Make predictions on the test set
y_pred = best_rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Visualize predictions vs. actual values. Limit to a reasonable number of samples for clarity.
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3) #draw a diagonal line
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs. Predicted Insurance Charges (Random Forest)")
plt.savefig("rf_predictions_vs_actual.png") #Save the plot!
print("✅ Scatter plot of predictions saved as 'rf_predictions_vs_actual.png'")


# Feature Importance (very helpful to understand the model)
feature_importances = best_rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
plt.title("Feature Importances (Random Forest)")
plt.savefig("rf_feature_importances.png")
print("✅ Feature importances plot saved as 'rf_feature_importances.png'")

####################################
# Load the dataset (replace with your actual file path)
file_path = "D:/University-Prj/ML0/insurance.csv"
insurance_data = pd.read_csv(file_path)

# Calculate average charges for smokers and non-smokers
avg_charges_smoker = insurance_data[insurance_data['smoker'] == 'yes']['charges'].mean()
avg_charges_nonsmoker = insurance_data[insurance_data['smoker'] == 'no']['charges'].mean()

# Calculate percentage difference
percentage_increase = ((avg_charges_smoker - avg_charges_nonsmoker) / avg_charges_nonsmoker) * 100

print(f"The average charges for smokers are {percentage_increase:.2f}% higher than for non-smokers.")

#Ensure 'smoker' column is in 'Yes'/'No' format
insurance_data['smoker'] = insurance_data['smoker'].map({'yes': 'Yes', 'no': 'No'})

# Create the bar plot
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
sns.barplot(x='smoker', y='charges', data=insurance_data, ci="sd", palette="viridis", order=['No', 'Yes']) #ci="sd" shows the standard deviation

plt.title("Average Medical Charges by Smoking Status")
plt.xlabel("Smoking Status")
plt.ylabel("Average Medical Charges")

#Save the plot to a file
plt.savefig("smoker_vs_charges_barplot.png")
plt.show()

print("✅ Bar plot of smoker vs. charges saved as 'smoker_vs_charges_barplot.png'")

###################

# Print feature importances
print("\nFeature Importances:")
print(importance_df)
#----------------------------------------------importing libraries------------------------------------------------#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer # <-- Added for handling missing data
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
#making a dataframe from the csv file
df = pd.read_csv("kepler_objects_of_interest.csv", comment="#")

#----------------------------------------------data preprocessing------------------------------------------------#

# Select relevant columns for the model
columns_to_keep = [
    'koi_period', 'koi_ror', 'koi_prad', 'koi_dor', 'koi_depth', 'koi_model_snr', 
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_fpflag_nt', 'koi_fpflag_ss', 
    'koi_fpflag_co', 'koi_fpflag_ec', 'koi_impact', 'koi_duration', 'koi_disposition'
]
processed_data = df[columns_to_keep].copy()

# Map target variable to numbers
processed_data["koi_disposition"] = processed_data["koi_disposition"].map({
    'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0
})

# Drop rows where the target variable itself is missing
processed_data.dropna(subset=['koi_disposition'], inplace=True)

# Define features (X) and target (Y)
X = processed_data.drop("koi_disposition", axis=1)
Y = processed_data["koi_disposition"]

# --- Impute missing values in features ---
print("Handling missing values using median imputation...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns) # Convert back to a DataFrame
print("Missing values handled.")


#----------------------------------------------model training------------------------------------------------#
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def tune_rf_hyperparameters(X_train, Y_train):
    param_grid = {
        'n_estimators': [100, 200],      # Number of trees in the forest
        'max_depth': [10, 20, None],     # Max depth of the tree
        'min_samples_split': [2, 5],     # Min samples required to split a node 
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        param_grid, 
        cv=3, 
        scoring='accuracy', 
        n_jobs=-1
    )
    grid_search.fit(X_train, Y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

print("\nTuning Random Forest hyperparameters... this may take a few minutes.")
best_rf_model = tune_rf_hyperparameters(X_train, Y_train)

#----------------------------------------------model evaluation------------------------------------------------#
def evaluate_model(model, X_test, Y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    matrix = confusion_matrix(Y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_rf_model, X_test, Y_test)
print(f"\nRandom Forest Model Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(matrix)

#----------------------------------------------visualizations------------------------------------------------#
def plot_confusion_matrix(matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'], yticklabels=['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Model with Imputation')
    plt.show()

print("\nDisplaying Confusion Matrix...")
plot_confusion_matrix(matrix)

#-----------------------------------------------feature importance visualization------------------------------------------------#
print("Displaying Feature Importance...")
importances = best_rf_model.feature_importances_
feature_names = X.columns

# Create a pandas Series for easy sorting and plotting
feature_importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot the results as a horizontal bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importance_series, y=feature_importance_series.index, palette='viridis')

# Add titles and labels for clarity
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance in Exoplanet Detection')
plt.tight_layout()
plt.show()
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data.csv")

# Map categorical health status labels to numerical values
health_status_mapping = {
    'Healthy': 0,
    'Very Healthy': 1,
    'Unhealthy': 2,
    'Sub-healthy': 3
}
df['Health_Status'] = df['Health_Status'].map(health_status_mapping)

# Separate features and target
X = df.drop(['Health_Status', 'Plot_ID', 'Latitude', 'Longitude'], axis=1)
Y = df['Health_Status'].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Apply SMOTE for handling class imbalance in the training data
oversampler = SMOTE(random_state=42)
x_train_smote, y_train_smote = oversampler.fit_resample(x_train, y_train)

# Neural Network Model Definition
nn_model = Sequential([
    Dense(units=16, activation='relu', input_dim=x_train_smote.shape[1]),
    Dense(units=32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(units=4, activation='softmax')  # 4 output units for multi-class classification
])

# Compile the Neural Network model
nn_model.compile(
    optimizer='adam',
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the Neural Network model
nn_history = nn_model.fit(
    x_train_smote, y_train_smote,
    validation_data=(x_test, y_test),
    epochs=100,
    verbose=1
)

# Random Forest Model Definition and Hyperparameter Tuning
rf_model = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [50, 100, 200, 300]}

# Grid Search with 5-fold cross-validation for optimal n_estimators
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train_smote, y_train_smote)
best_n_estimators = grid_search.best_params_['n_estimators']

# Train the Random Forest model with optimal parameters
rf_model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
rf_model.fit(x_train_smote, y_train_smote)

# XGBoost Model Definition and Training
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(x_train_smote, y_train_smote)

# Displaying Models' Performance
print("Neural Network Model Performance on Test Data:")
print(classification_report(y_test, np.argmax(nn_model.predict(x_test), axis=1)))

print("Random Forest Model Performance on Test Data:")
print(classification_report(y_test, rf_model.predict(x_test)))

print("XGBoost Model Performance on Test Data:")
print(classification_report(y_test, xgb_model.predict(x_test)))
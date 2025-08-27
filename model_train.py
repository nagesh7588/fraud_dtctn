import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv('dataset.csv')
X = df.drop('is_fraud', axis=1)
X = pd.get_dummies(X)
y = df['is_fraud']

# Train model

# Save feature names
feature_names = list(X.columns)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and feature names
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

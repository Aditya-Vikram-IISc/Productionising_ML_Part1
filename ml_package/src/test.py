from .utils import DataReader, ConfigReader
import os

PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PATH, "configs/config.yaml")
# Read the configurations
configs = ConfigReader(path = CONFIG_PATH).getConfigParams()
print(configs["all_models"])
models = {m: all__models[m] for m in list(configs["all_models"].keys())}
print(models)



import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

# Step 1: Generate a synthetic dataset
np.random.seed(0)
data_size = 1000

# Create a DataFrame with numerical and categorical features
data = pd.DataFrame({
    'age': np.random.randint(18, 70, size=data_size),
    'income': np.random.randint(20000, 100000, size=data_size),
    'gender': np.random.choice(['male', 'female'], size=data_size),
    'occupation': np.random.choice(['engineer', 'doctor', 'artist', 'teacher'], size=data_size),
    'target': np.random.choice([0, 1], size=data_size)
})

# Introduce missing values
data.loc[np.random.choice(data.index, size=100, replace=False), 'income'] = np.nan

# Separate features and target
X = data.drop(columns='target')
y = data['target']

# Step 2: Define preprocessing steps
numerical_features = ['age', 'income']
categorical_features = ['gender', 'occupation']

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps into a single column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Step 3: Define models and parameter grids
models = {
    'RandomForest': RandomForestClassifier(),
    'SVC': SVC(),
    'LogisticRegression': LogisticRegression()
}

param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'SVC': {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__kernel': ['linear', 'rbf']
    },
    'LogisticRegression': {
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['liblinear']
    }
}

# Step 4: Setup and perform Grid Search
best_pipeline = None
best_score = -np.inf
best_model_name = None

for model_name, model in models.items():
    # Ensure that model is an instance of an estimator, not a string
    if not isinstance(model, (RandomForestClassifier, SVC, LogisticRegression)):
        raise TypeError(f"Expected an estimator, got {type(model).__name__} instead.")
    
    # Define the pipeline with the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Get the corresponding parameter grid
    param_grid = param_grids.get(model_name, {})
    if not param_grid:
        raise ValueError(f"No parameter grid found for model {model_name}.")

    # Perform Grid Search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X, y)
    
    # Capture results
    mean_test_score = grid_search.best_score_
    print(f"Model: {model_name}")
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Best Score: {mean_test_score:.4f}")
    
    # Check if this model is the best so far
    if mean_test_score > best_score:
        best_score = mean_test_score
        best_pipeline = grid_search.best_estimator_
        best_model_name = model_name

# Save the best pipeline
if best_pipeline:
    print(f"Saving the best pipeline: {best_model_name} with score: {best_score:.4f}")
    joblib.dump(best_pipeline, 'best_pipeline.pkl')
else:
    print("No model found.")

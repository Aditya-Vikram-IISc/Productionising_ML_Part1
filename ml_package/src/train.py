import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from .model_pipeline import Preprocessor
from .utils import DataReader, ConfigReader


PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PATH, "configs/config.yaml")

all__models = {
                "random_forest": RandomForestClassifier(),
                "svc": SVC(),
                "logisticregression": LogisticRegression()
               }


def train():
    # Read the configurations
    configs = ConfigReader(path = CONFIG_PATH).getConfigParams()

    # Load data
    df = DataReader(path = os.path.join(PATH, configs["data_path"]["train_data"])).get_df()

    # Split the data into train test
    X_train, X_test, y_train, y_test = train_test_split(df[df.columns.drop('Survived')], df["Survived"],
                                                        test_size = configs["train_hyperparameters"]["train_test_split"],
                                                        random_state = 0
                                                        )
    
    # Initialize and configure the Preprocessor Pipeline
    dataprocessor_pipeline = Preprocessor(features_to_drop = configs["all_features"]["drop_columns"],
                            numerical_columns = configs["all_features"]["numerical_features"],
                            numerical_column_impute_strategy = "mean",
                            categorical_column = configs["all_features"]["categotical_features"],
                            categorical_column_impute_strategy = "mode",
                            categorical_column_encode_strategy = "by_countorder"
                            )
    
    df = dataprocessor_pipeline.fit_transform(X_train, y_train)

    # Define the models - Key value pair of model_identifier & model_instance
    models = {m: all__models[m] for m in list(configs["all_models"].keys())}
    param_grid = configs["all_models"]
    

    # Setup and perform Grid Search
    best_pipeline = None
    best_score = -np.inf
    best_model_name = None
    results = {}

    for model_name, model in models.items():

        pipeline = Pipeline(steps = [
                            ("preprocessor", dataprocessor_pipeline),
                            ("classifier", model)
        ])

        param_grid = param_grid[model_name]
        grid_search = GridSearchCV(pipeline, param_grid, cv=configs["train_hyperparameters"]["kfold"],
                                    n_jobs=-1, verbose=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        #Capture Results
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
        joblib.dump(best_pipeline, os.path.join(PATH, 'trained_model/best_pipeline.pkl'))
    else:
        print("No model found.")
    

if __name__ == "__main__":
    train()
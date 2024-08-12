import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from .model_pipeline import Preprocessor
from .utils import DataReader, ConfigReader


PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PATH, "configs/config.yaml")


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
    data_pipeline = Preprocessor(features_to_drop = configs["all_features"]["drop_columns"],
                            numerical_columns = configs["all_features"]["numerical_features"],
                            numerical_column_impute_strategy = "mean",
                            categorical_column = configs["all_features"]["categotical_features"],
                            categorical_column_impute_strategy = "mode",
                            categorical_column_encode_strategy = "by_countorder"
                            )
    
    # Define the models 
    


    

    
    



if __name__ == "__main__":
    train()
import pandas as pd
import yaml
import os

all_classes = ["DataReader", "ConfigReader"]

class DataReader:
    """DataReader for reading the pandas dataframe.

    Args:
        path (str): Path of the csv file from where dataframe is loaded
    """
    def __init__(self, path):
        self.path = path

        try:
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"No file exist at the given path {self.path}")
            
            self.df = pd.read_csv(self.path)

        except FileNotFoundError as fnfe:
            raise fnfe
    
    def __repr__(self):
        return f"Dataframe loaded from path {self.path}"
    

    def get_columns(self):
        return list(self.df.columns)
    
    def get_df(self):
        return self.df



class ConfigReader:
    """Config Reader for reading configurations across data sources, model_features,  training_hyperparameters etc.

    Args:
        path (str): Path of the dataframe
    """
    def __init__(self, path:str):
        self.path = path
        
        try:
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"The file at path {self.path} does not exist.")
            
            with open(self.path, "r") as f:
                self.configparams = yaml.load(f, Loader= yaml.SafeLoader)

        except FileNotFoundError as fe:
            raise fe
        
        except yaml.YAMLError as ye:
            raise ValueError(f"Error parsing YAML file: {ye}")
    
    def __repr__(self):
        return f"Config file taken from path: {self.path}"

    def getConfigParams(self):
        return self.configparams


if __name__ == "__main__":
    PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    #  Check for ConfigReader
    config_path = os.path.join(PATH, "configs\config.yaml")
    config_ = ConfigReader(config_path).getConfigParams()

    #  Check for DataReader
    df = DataReader(os.path.join(PATH, config_["data_path"]["train_data"]))
    print(df.get_df().columns)
import pandas as pd
import os
from .utils import DataReader, ConfigReader, load_pipeline

# get repo folder path
PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# read the config
CONFIG_PATH = os.path.join(PATH, "configs/config.yaml")
configs = ConfigReader(CONFIG_PATH).getConfigParams()

# Get the pipeline and data path
PIPELINE_PATH = os.path.join(PATH,f'{configs["model_path"]["model_folderpath"]}/best_pipeline.pkl')
DATA_PATH = os.path.join(PATH, configs["data_path"]["test_data"])

def generate_predicitons(data_path:str, pipeline_path:str) -> list:
    # load the dataset
    df =DataReader(data_path).get_df()    

    # Load the ML pipeline
    model_pipeline = load_pipeline(pipeline_path)
    
    # get output
    op = model_pipeline.predict(df)
    return list(op)

if __name__ == "__main__":
    op = generate_predicitons(DATA_PATH, PIPELINE_PATH)
    print(op[:25])
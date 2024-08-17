import os
import numpy as np
from ml_package.src.predict import generate_predicitons
from ml_package.src.utils import DataReader, ConfigReader, load_pipeline
from ml_package.src.predict import generate_predicitons


# get repo folder path
PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_package")

# read the config
CONFIG_PATH = os.path.join(PATH, "configs/config.yaml")
configs = ConfigReader(CONFIG_PATH).getConfigParams()


# Get the pipeline and data path
PIPELINE_PATH = os.path.join(PATH,f'{configs["model_path"]["model_folderpath"]}/best_pipeline.pkl')
DATA_PATH = os.path.join(PATH, configs["data_path"]["test_data"])



def test_prediction():
    # Load the daatset
    df =DataReader(DATA_PATH).get_df() 

    # Load the ML pipeline
    model_pipeline = load_pipeline(PIPELINE_PATH)

    # When
    output = model_pipeline.predict(df.iloc[0:1])

    # Then
    assert output is not None
    assert isinstance(output[0], (int, np.int64)), "Output value is not np.int64"
    assert int(output[0]) in (0,1), "Output value is not among 0 and 1"
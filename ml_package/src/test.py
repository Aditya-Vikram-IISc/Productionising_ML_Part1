from .utils import DataReader, ConfigReader
import os

PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PATH, "configs/config.yaml")
# Read the configurations
configs = ConfigReader(path = CONFIG_PATH).getConfigParams()
print([k for k in configs["all_models"].keys()])

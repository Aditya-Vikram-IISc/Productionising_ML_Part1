import io
import os
from pathlib import Path
from setuptools import find_packages, setup


# Package meta-data.
NAME = 'ml_package'
DESCRIPTION = 'Stepping stone towards productioning ML solution'
URL = 'https://github.com/Aditya-Vikram-IISc/Productionising_ML_Part1.git'
EMAIL = 'aditya.vikram@email.com'
AUTHOR = 'Aditya Vikram'
REQUIRES_PYTHON = '>=3.10.8'


# What packages are required for this module to be executed?
def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


# If you do change the License, remember to change the Trove Classifier for that!
here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
about = {}
with open(ROOT_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version



# Lets pack teh package :) :
setup(
    name=NAME,                                          # Name of your package
    version=about['__version__'],                       # Version of the package
    description="A machine learning model package",     # Short description
    long_description=long_description, 
    long_description_content_type='text/markdown',
    author=AUTHOR,                                      # Your name
    author_email=EMAIL,                                 # Your email
    python_requires=REQUIRES_PYTHON,                    # Python version requirement
    url=URL,                                            # GitHub repo URL
    packages=find_packages(exclude=('tests',)),
    package_data={'regression_model': ['VERSION']},
    install_requires=list_reqs(),                       # Install dependencies basis the requirement.txt
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3.10.8",
        'Operating System :: OS Independent'
    ],
)
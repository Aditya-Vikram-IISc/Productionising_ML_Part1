# Productionising_ML_Part1
A guide for budding ML Scientists to learn how to write a production level ML code. Link to medium article to be shared soon.
#### 1. Project Folder Structure

The project folder structure is as follows:

![Project Structure](images/Image1.png)


* MANIFEST.in : Added especially if you want to include additional files (like README.md, LICENSE, VERSION, etc.) in your source distribution that are not automatically included by setuptools.
<br/><br/>
* LICENCE : Licence as per your criteria i.e MIT, Apache etc
<br><br>
* LICENCE : Licence as per your criteria i.e MIT, Apache etc
<br/><br/>
* requirements.txt : Contains external packages required for our package to work successfuly
<br/><br/>
* requirements_dev.txt : Contains all packages mentioned in requirements.txt plus pytest, setuptools, and wheel required to create the package. We will create our local working environment using requirements_dev.txt
<br/><br/>
* README.md : For project description
<br/><br/>
* setup.py : This script is essential for packaging
<br/><br/>
* VERSION : Contains the version mentioned in  `MAJOR.MINOR.PATCH`
<br/><br/>

#### 2. Instructions to create the package

* Ensure that your pwd is PRODUCTIONING_ML_PART1. Also, ensure that your have created all the nessesary files as mentioned above
<br/><br/>
* Ti run the pytest modules, type teh following in terminal
`pytest tests/`
<br/><br/>
* To create the wheel file, type the following
`python setup.py sdist bdist_wheel`
<br/><br/>
* Once this code is executed two new folders get created namely ***build*** and ***dist*** and the folder structure is as shows
<br/><br/>

![Project Output Structure](images/Image2.png)

<br/><br/>
* Now to install the package from the wheel file as follows.
`pip install dist/ml_package-1.1.0-py3-none-any.whl`
<br/><br/>
* One can now simply import all functionalities as
`from ml_package import train, ConfigReader etc.`

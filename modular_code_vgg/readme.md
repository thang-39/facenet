# Execution Instructions
Note: It is advisable not to run this script on Google Colab due to the following reasons:

Python Version Compatibility: Google Colab has changed its default Python version to 3.10.12, while the script was developed on earlier versions. This discrepancy in Python versions may lead to compatibility issues and unexpected errors. To ensure compatibility, it is recommended to create a virtual environment locally with Python 3.7.0 for VGG, as specified in the execution instructions.


# Python version 3.7.0

To create a virtual environment and install requirements in Python 3.6.2 on different operating systems, follow the instructions below:

### For Windows:

Open the Command Prompt by pressing Win + R, typing "cmd", and pressing Enter.

Change the directory to the desired location for your project:

`cd C:\path\to\project`

Create a new virtual environment using the venv module:

`python -m venv myenv`

Activate the virtual environment:

`myenv\Scripts\activate`

Install the project requirements using pip:

`pip install -r requirements.txt`


### For Linux/Mac:

Open a terminal.

Change the directory to the desired location for your project:

`cd /path/to/project`

Create a new virtual environment using the venv module:

`python3 -m venv myenv`

Activate the virtual environment:

`source myenv/bin/activate`

Install the project requirements using pip:

`pip install -r requirements.txt`

These instructions assume you have Python 3.7.0 installed and added to your system's PATH variable.


## Execution Instructions if Multiple Python Versions Installed


If you have multiple Python versions installed on your system , you can use the Python Launcher to create a virtual environment with Python 3.7.0, you can specify the version using the -p or --python flag. Follow the instructions below:

### For Windows:

Open the Command Prompt by pressing Win + R, typing "cmd", and pressing Enter.

Change the directory to the desired location for your project:

`cd C:\path\to\project`

Create a new virtual environment using the Python Launcher:

`py -3.7 -m venv myenv`

Note: Replace myenv with your desired virtual environment name.

Activate the virtual environment:

`myenv\Scripts\activate`

Install the project requirements using pip:

`pip install -r requirements.txt`


For Linux/Mac:

Open a terminal.

Change the directory to the desired location for your project:

`cd /path/to/project`

Create a new virtual environment using the Python Launcher:

`python3.7 -m venv myenv`
Note: Replace myenv with your desired virtual environment name.

Activate the virtual environment:

`source myenv/bin/activate`
Install the project requirements using pip:

`pip install -r requirements.txt`


By specifying the version using py -3.7 or python3.7, you can ensure that the virtual environment is created using Python 3.7.0 specifically, even if you have other Python versions installed.







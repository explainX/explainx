<h1 align="center">
	<img width="300" src="https://i.ibb.co/yY7tfDg/Logo.jpg" alt="explainX.ai"> 
	<br>
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.6%20|%203.7|%203.8-brightgreen.svg" alt="Python supported"></a>
   <a href="https://pypi.org/project/explainx/"><img src="https://badge.fury.io/py/explainx.svg" alt="PyPi Version"></a>
  <!-- <a href="https://pypi.org/project/explainx/"><img src="https://img.shields.io/pypi/dm/explainx" alt="PyPi Downloads"></a> -->
  <a href="https://www.explainx.ai/"><img src="https://img.shields.io/website?url=https%3A%2F%2Fwww.explainx.ai%2F" alt="explainx.ai website"></a>
</p>


ExplainX.ai is a fast, light-weight and scalable Explainable AI framework for data scientists. It enables you to explain and debug state of the art machine learning models in as simple as one line of code. [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Explain%20any%20black-box%20Machine%20Learning%20model%20in%20just%20one%20line%20of%20code%21&url=https://www.explainx.ai&hashtags=xai,explainable_ai,explainable_machine_learning,trust_in_ai,transparent_ai)

<img width="600" src="https://i.ibb.co/w4SF1GJ/Group-2-1.png" alt="explainX.ai">

#### Why we need explainability & interpretibility?

Essential for:
1. Model debugging - Why did my model make a mistake? How can I improve the accuracy of the model?
2. Detecting fairness issues - Is my model biased? If yes, where?
3. Human-AI cooperation - How can I understand and trust the model's decisions?
4. Regulatory compliance - Does my model satisfy legal & regulatory requirements?
5. High-risk applications - Healthcare, Financial Services, FinTech, Judicial, Security etc,.

Visit explainx.ai website to learn more: https://www.explainx.ai     



## Try it out

* [Installing explainX](https://explainx-documentation.netlify.app/)
* [Working Examples](https://explainx-documentation.netlify.app/working-example/)
* [explainX Dashboard Features](https://explainx-documentation.netlify.app/analyze-dashboard/)
* [Documentation](https://explainx-documentation.netlify.app/)
* [Provide Feedback to Improve explainX.ai](https://forms.gle/5Q1xaHd7s6UQkRzf8)

## Installation

* **Desktop**: You can use explainX on your own computer in under a minute. If you already have a python environment setup, just run the following command.

* Make sure you have **Python 3.5+**
* Looking to run **explainX** on the **cloud**? Install **nodejs** and **localtunnel** using the following instructions.

### To install **nodejs** and **localtunnel** on **MAC OS**
* Open the terminal.
* Install Xcode Command Line Tools using the following.
```python
xcode-select --install
```
* Install **brew** using the following.
```python
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" 
```
* Install **nodejs** using the following.
```python
brew install nodejs
```
* Install **localtunnel** using the following.
```python
npm install -g localtunnel
```
### To install **nodejs** and **localtunnel** on **Ubuntu**
* Open the terminal.
* Install **nodejs** using the following.
```python
sudo apt install nodejs
```
* Install **npm** using the following.
```python
sudo apt install npm
```
* Install **localtunnel** using the following.
```python
npm install -g localtunnel
```
### To install **nodejs** and **localtunnel** on **CentOS**
* Open the terminal.
* Run the following command.
```python
curl -sL https://rpm.nodesource.com/setup_10.x | sudo bash -
```
* Install **nodejs** using the following.
```python
sudo yum install nodejs
```
* Install **npm** using the following.
```python
sudo yum install npm
```
* Install **localtunnel** using the following.
```python
npm install -g localtunnel
```
### To install **nodejs** and **localtunnel** on **Windows**
* Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) found [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
* Install [Nodejs](https://nodejs.org/en/download/) found [here](https://nodejs.org/en/download/).
* Open the terminal and run the following to install **localtunnel**.
```python
npm install -g localtunnel
```

### Lastly, install **ExplainX** using the following.

```python
pip install explainx
```
* **Jupyter Notebook**: You can also install explainx via Jupyter Notebook. Just run the following command:

```python
!pip install explainx
```

## Usage

Once you have install explainX, you can simply follow the example below to use it:

Import **explainx**

```python
from explainx import *
```

Load dataset as X_Data, Y_Data in your XGBoost Model

```python
#X_Data = Pandas DataFrame
#Y_Data = Numpy Array or List

X_Data, Y_Data = explainx.dataset_boston()

#Train Model
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X_Data, label=Y_Data), 100)
```

One line of code to **use the explainx module**

```python
explainx.ai(X_Data, Y_Data, model, model_name="xgboost")
```

Click on the link to view the dashboard.

```jupyter
App running on https://0.0.0.0:8080
```
Running ExplainX on the **cloud e.g., AWS Sagemaker?** If **https://0.0.0.0:8080** does not work, open the **terminal** and run the following command.

```jupyter
lt -h "https://serverless.social" -p [port number]
```
```jupyter
lt -h "https://serverless.social" -p 8080
```
Learn to analyze the dashboard by following this link: [explainX Dashboard Features](https://explainx-documentation.netlify.app/analyze-dashboard/)

Visit the documentation to [learn more](https://explainx-documentation.netlify.app/)

## Models Supported
CatBoost, XGBoost, Scikit-learn Models, SVM, Neural Networks


## Video Tutorial

Please click on the image below to load the tutorial:

[![here](https://github.com/explainX/explainx/blob/master/explain_video_img.png)](https://youtu.be/X3fk-r2G15k)  

(Note: Please manually set it to 720p or greater to have the text appear clearly)

## Contributing
Pull requests are welcome. In order to make changes to explainx, the ideal approach is to fork the repository then clone the fork locally.

For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## Report Issues

Please help us by [reporting any issues](https://github.com/explainX/explainx/issues/new) you may have while using explainX.

## License
[MIT](https://choosealicense.com/licenses/mit/)

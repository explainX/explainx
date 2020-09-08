<h1 align="center">
	<img width="700" src="main_page_banner.png" alt="explainX.ai"> 
	<br>
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.6%20|%203.7|%203.8-brightgreen.svg" alt="Python supported"></a>
  <!-- <a href="https://pypi.org/project/explainx/"><img src="https://badge.fury.io/py/explainx.svg" alt="PyPi Version"></a> -->
  <!-- <a href="https://pypi.org/project/explainx/"><img src="https://img.shields.io/pypi/dm/explainx" alt="PyPi Downloads"></a> -->
  <a href="https://www.explainx.ai/"> <img src="https://img.shields.io/website?url=https%3A%2F%2Fwww.explainx.ai%2F" alt="explainx.ai website"></a>
</p>

ExplainX.ai is a fast, scalable and end-to-end Explainable AI framework for data scientists & machine learning engineers. 

With explainX, you can understand overall model behavior, get the reasoning behind model predictions, remove biases and create convincing explanations for your business stakeholders. [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Explain%20any%20black-box%20Machine%20Learning%20model%20in%20just%20one%20line%20of%20code%21&url=https://www.explainx.ai&hashtags=xai,explainable_ai,explainable_machine_learning,trust_in_ai,transparent_ai)

<img width="800" src="started_example.png" alt="explainX.ai">

#### Why we need explainability & interpretibility?

Essential for:
1. Model debugging - Why did my model make a mistake? How can I improve the accuracy of the model?
2. Detecting fairness issues - Is my model biased? If yes, where?
3. Human-AI cooperation - How can I understand and trust the model's decisions?
4. Regulatory compliance - Does my model satisfy legal & regulatory requirements?
5. High-risk applications - Healthcare, Financial Services, FinTech, Judicial, Security etc,.

Visit our website to learn more: https://www.explainx.ai     

## Try it out

* [Installing explainX](https://docs.explainx.ai/getting-started/installation)
* [Working Examples](https://docs.explainx.ai/getting-started/starter-example)
* [explainX Dashboard Features](https://docs.explainx.ai/tutorials/analyzing-dashboard)
* [Documentation](https://docs.explainx.ai/)
* [Help Us Improve explainX.ai](https://forms.gle/5Q1xaHd7s6UQkRzf8)


# Installation

Python 3.5+ | Linux, Mac, Windows (Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first to run on Windows.)

```sh
pip install explainx
```

## Installation on the cloud
If you are using AWS SageMaker of Jupyter Notebook deployed on the cloud, visit our documentation for step-by-step guide installing and running explainX. [Cloud Installation Instructions](https://docs.explainx.ai/getting-started/installation)


## Example Usage
After successfully installing explainX, open up your Python IDE of Jupyter Notebook and simply follow the code below to use it:

1. Import **explainx** module.

```python
from explainx import *
```

2. Load and split your dataset into x_data and y_data

```python
#x_data = Pandas DataFrame
#y_data = Numpy Array or List

x_data, y_data = explainx.dataset_boston() 
```

3. Train your model.

```python
#Train Model
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(x_data, label=y_data), 100)
```

4. Pass your model and dataset into the explainX function:

```python
explainx.ai(x_data, y_data, model, model_name="xgboost")
```

5. Click on the dashboard link to start exploring model behavior:

```python
App running on https://0.0.0.0:8080
```

**If you are running explainX on the cloud e.g., AWS Sagemaker?** **https://0.0.0.0:8080** will not work.
Please visit our documentation for installation instructions for the cloud: [Cloud Installation Instructions](https://docs.explainx.ai/getting-started/installation)

After installation is complete, just open your **terminal** and run the following command.
```jupyter
lt -h "https://serverless.social" -p [port number]
```
```jupyter
lt -h "https://serverless.social" -p 8080
```

<img width="1000" src="demo-explainx-with-sound.gif" alt="explainX.ai">

Learn to analyze the dashboard by following this link: [explainX Dashboard Features](https://explainx-documentation.netlify.app/analyze-dashboard/)

Visit the documentation to [learn more](https://docs.explainx.ai/)

## Models Supported
1. Catboost
2. xgboost==1.0.2
3. Gradient Boosting Regressor
4. RandomForest Model
5. SVM
6. KNeighboursClassifier
7. Logistic Regression
8. DecisionTreeClassifier
9. Scikit-learn Models
10. Neural Networks

## Walkthough Video Tutorial

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

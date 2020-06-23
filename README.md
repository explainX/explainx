<h1 align="center">
	<img width="300" src="https://i.ibb.co/yY7tfDg/Logo.jpg" alt="explainX.ai"> 
	<br>
</h1>

ExplainX.ai is a fast, scalable & state-of-the-art explainable AI platform. ExplainX.ai helps data scientists understand, explain, debug and validate any machine learning model - in just one line of code.

[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Explain%20any%20black-box%20Machine%20Learning%20model%20in%20just%20one%20line%20of%20code%21&url=https://www.explainx.ai&hashtags=xai,explainable_ai,explainable_machine_learning,trust_in_ai,transparent_ai)

#### Why we need explainability & interpretibility?

Essential for:
1. Model debugging - Why did my model make a mistake? How can I improve the accuracy of the model?
2. Detecting fairness issues - Is my model biased? If yes, where?
3. Human-AI cooperation - How can I understand and trust the model's decisions?
4. Regulatory compliance - Does my model satisfy legal & regulatory requirements?
5. High-risk applications - Healthcare, Financial Services, FinTech, Judicial, Security etc,.

Visit explainx.ai website to learn more: https://www.explainx.ai     
<a href="https://www.explainx.ai/"><img src="https://img.shields.io/website?url=https%3A%2F%2Fwww.explainx.ai%2F" alt="explainx.ai website"></a>

<img width="600" src="https://i.ibb.co/w4SF1GJ/Group-2-1.png" alt="explainX.ai">


## Try it out

* [Installing explainX](https://explainx-documentation.netlify.app/)
* [Working Examples](https://explainx-documentation.netlify.app/working-example/)
* [explainX Dashboard Features](https://explainx-documentation.netlify.app/analyze-dashboard/)
* [Documentation](https://explainx-documentation.netlify.app/)
* [Provide Feedback to Improve explainX.ai](https://forms.gle/5Q1xaHd7s6UQkRzf8)

### Installation

* **Desktop**: You can use explainX on your own computer in under a minute. If you already have a python environment setup, just run the following command.

```python
pip install explainx
```
* **Jupyter Notebook**: You can also install explainx via Jupyter Notebook. Just run the following command:

```python
!pip install explainx
```

### Usage

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

Click on the link to view the dashboard:

```jupyter
App running on https://127.0.0.1:8050
```

Learn to analyze the dashboard by following this link: [explainX Dashboard Features](https://explainx-documentation.netlify.app/analyze-dashboard/)

Visit the documentation to [learn more](https://explainx-documentation.netlify.app/)

### Models Supported
CatBoost, XGBoost, Scikit-learn Models, SVM, Neural Networks

## Video Tutorial

Please click on the image below to load the tutorial.

## Contributing
Pull requests are welcome. In order to make changes to explainx, the ideal approach is to fork the repository than clone the fork locally.

For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## Report Issues

Please help us by [reporting any issues](https://github.com/explainX/explainx/issues/new) you may have while using explainX.

## License
[MIT](https://choosealicense.com/licenses/mit/)

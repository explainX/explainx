<h1 align="center">
	<img width="300" src="https://i.ibb.co/yY7tfDg/Logo.jpg" alt="explainX.ai"> 
	<br>
</h1>


explainX.ai helps data scientists understand, explain and validate any machine learning model - in just one line of code.

[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Explain%20any%20black-box%20Machine%20Learning%20model%20in%20just%20one%20line%20of%20code%21&url=https://www.explainx.ai&hashtags=xai,explainable_ai,explainable_machine_learning,trust_in_ai,transparent_ai)

Visit explainx.ai website to learn more; https://www.explainx.ai

## Try it out

* [Installing explainX](https://explainx-documentation.netlify.app/)
* [Working Example](https://explainx-documentation.netlify.app/working-example/)
* [explainX Dashboard Features](https://explainx-documentation.netlify.app/analyze-dashboard/t)

### Installation

* **Desktop**: You can use explainX in your Jupyter Notebook in under a minute. Just run the following command in your code block.

```python
!pip install explainx
```

### Usage

Once you have install explainX, you can simply follow the example below to use it:

Import **explainx**

```python
from explainx import *
```

Load Dataset & pass X_Data, Y_Data as numpy arrays in your XGBoost Model

```python
X_data, Y_data = explainx.dataset_boston()

model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=Y_data), 100)
```

One line of code to **use the explainx module**

```python
explainx.ai(X_Data, Y_Data, model, model_name="xgboost")
```

Click on the link to view the dashboard

```jupyter
App running on https://127.0.0.1:8050
```


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

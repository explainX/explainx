# explainX

explainX.ai helps data scientists understand, explain and validate any machine learning model - in just one line of code. Checkout explainx.ai to learn more.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install explainx``

## Usage

#Import the library
from explainx import *

#Load Dataset
X_data, Y_data = explainx.dataset_boston()

#Pass X_data, Y_data as numpy arrays into your XGBoost Model
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=Y_data), 100)

#Pass your X_data, Y_data, y_variable name, model and model name to the explainx function
explainx.ai(X_data, Y_data, model, model_name="xgboost")

#Click on the link to access the dashboard
App running on https://127.0.0.1:8050
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
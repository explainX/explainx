from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
import dash_core_components
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs.scatter.marker import Line
import plotly.figure_factory as ff
import dash_table
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import time
import shap
import dash_editor_components
import socket
from contextlib import closing
import subprocess
import os
import sys
import shap
import random
import string
import h2o
from uuid import getnode as get_mac
import platform
import datetime
import pyrebase
from config_det import data_det
from collections import deque
from sklearn import metrics
from sklearn.base import is_classifier, is_regressor
import pytest

firebase_app = pyrebase.initialize_app(data_det)
ref = firebase_app.database()

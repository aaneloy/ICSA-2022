import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.datasets import mnist, fashion_mnist

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
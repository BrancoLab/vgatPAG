import numpy as np
import math
import pandas as pd
import time
import glob


from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import mixture, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def boxcar_center(a, n):
  a1 = pd.Series(a)
  moving_avg = np.array(a1.rolling(window = n,min_periods=1).mean(center=True))
  return moving_avg 
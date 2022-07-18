

import numpy as np
import seaborn as sns
import pandas as pd

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

import matplotlib.pyplot as plt
sns.set_theme(color_codes=True)


sns.regplot(x="EstimatedSalary",y='Age', data=dataset)

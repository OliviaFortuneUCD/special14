import numpy as np
import seaborn as sns
import pandas as pd

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')



sns.heatmap(dataset.corr(),linecolor='white',linewidths=2,annot=True)



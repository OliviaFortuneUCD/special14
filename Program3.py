import numpy as np
import seaborn as sns
import pandas as pd

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')


sns.relplot(y="EstimatedSalary", x="Age", col="Gender",

           data=dataset)
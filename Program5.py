import seaborn as sns
from scipy import stats
import pandas as pd
x_col = "EstimatedSalary"
y_col = "Age"
hue_col = "Purchased"



#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

g = sns.jointplot(data=dataset, x=x_col, y=y_col, hue=hue_col)

for _,gr in dataset.groupby(hue_col):
    sns.regplot(x=x_col, y=y_col, data=gr, scatter=False, ax=g.ax_joint, truncate=False)
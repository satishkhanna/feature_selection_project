# Default imports
import pandas as pd
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv('data/house_prices_multivariate.csv')

def plot_corr (df,size = 11):
    subplots(figsize=(size,5))
    sns.heatmap(data.corr(),cmap = 'YlOrRd')
    plt.show()
# Write your solution here:

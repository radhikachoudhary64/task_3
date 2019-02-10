# task_3
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set(style="whitegrid", palette="muted")
tips = sns.load_dataset("tips")


f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)

sns.scatterplot(x="tip", y="total_bill",
                hue="day", size="size",
                palette="ch:r=-.3,d=4_r",
                hue_order=clarity_ranking,
                sizes=(1, 8), linewidth=0,
                data=tips, ax=ax)
                
                
                

ts = np.random.RandomState(244)
values = ts.randn(100, 4).cumsum(axis=0)
tips = pd.date_range("2.9 1  10", periods=10, freq="A")
data = pd.DataFrame(values, dates, columns=["thur", "fri", "sat", "sun"])
data = data.rolling(3).mean()
sns.lineplot(data=data, palette="tab10", linewidth=2.5)


ts = np.random.RandomState(7)
x = ts.normal(2, 1, 75)
y = 2 + 1.5 * x + ts.normal(0, 2, 75)




# Plot the residuals after fitting a linear model
sns.residplot(x, y, lowess=True, color="g")

t = np.linspace(0, 10, num=100)
df = pd.DataFrame({'t': t, 'thur': t, 'fri': 2 * t,  'sat': 3 * t,'sun':4*t})

# Convert the dataframe to long-form or "tidy" format
df = pd.melt(df, id_vars=['t'], var_name='day', value_name='tip')

# Set up a grid of axes with a polar projection
g = sns.FacetGrid(df, col="day", hue="tip",
                  subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)

# Draw a scatterplot onto each axes in the grid
g.map(sns.scatterplot, "tip", "t")



fig, ax = plt.subplots()
tips = pd.read_csv("tips.csv")
ax.violinplot(tips["total_bill"], vert=False)
plt.show()
 

tips = sns.load_dataset("tips") 
sns.violinplot(x="total_bill", data=tips)
plt.show()

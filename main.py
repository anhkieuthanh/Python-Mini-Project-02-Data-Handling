import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_raw = pd.read_csv('data.csv')

sexQty = data_raw.groupby('Sex').size()
print(sexQty)

plt.bar(sexQty.index, sexQty.values)
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Count of Passengers by Sex')
plt.xticks([0, 1], ['Female', 'Male'])
plt.show()

maleSurvivedRate = data_raw[data_raw['Sex'] == 1]['Survived'].mean()
femaleSurvivedRate = data_raw[data_raw['Sex'] == 0]['Survived'].mean()

plt.bar(['Male', 'Female'], [maleSurvivedRate, femaleSurvivedRate])
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Sex')
plt.ylim(0, 1)
plt.show()

#Heatmap correlation để soi feature nào quan trọng.
corr_matrix = data_raw.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
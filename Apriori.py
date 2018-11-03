import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/Apriori-Python/Apriori_Python/Market_Basket_Optimisation.csv', header = None)

'''We are converting the dataset to list of lists.'''

transactions = []
for i in range(0, len(data)):
    temp = [] 
    for j in  range(0, 20):
        temp.append(str(data.iloc[i,j]))
    transactions.append(temp)
    
'''
We have to support min_support, min_confidence, min_left, min_length =2(generally)
as we need at least 2 models for association.
'''

from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2 )

results = list(rules)
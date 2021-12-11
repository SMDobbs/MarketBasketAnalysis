import pandas as pd
import numpy as np
import sys
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Reading in the dataset which was obtained from Kaggle.com
data = pd.read_csv('TransactionData.csv')

data['Description'] = data['Description'].str.strip()
data.dropna(axis=0, subset=['Invoice'], inplace=True)
data['Invoice'] = data['Invoice'].astype('str')
data = data[~data['Invoice'].str.contains('C')]

#Here I chose to filter down the dataset to Germany and France to provide a large enough sample size but optimize run speed
mybasket = (data[(data['Country'] == "Germany") | (data['Country'] == "France")].groupby(['Invoice', 'Description'])['Quantity'].sum().unstack()
            .reset_index().fillna(0).set_index('Invoice'))

def ConvertToBoo(x):
    if x <= 0: 
        return 0
    if x >= 1:
        return 1

basketsets = mybasket.applymap(ConvertToBoo)
basketsets.drop('POSTAGE', inplace = True, axis=1)

# A Support =  % of transactions that have A

# C Support = % of transactions that have C

# Support = % of transactions that have A and C

# confidence(A→C)= support(A→C) / support(A),range: [0,1]  (1 would mean that everytime item A is purchased so is C)

#lift = How much more often they occur together than if they were statistically independent. If independent lift = 1

# Leverage computes the difference between the observed frequency of A and C appearing together and the 
#frequency that would be expected if A and C were independent. 
#A leverage value of 0 indicates independence.

#A high conviction value means that the consequent is highly depending on the antecedent. For instance,
#in the case of a perfect confidence score, the denominator becomes 0 (due to 1 - 1) for which the conviction score
#is defined as 'inf'. Similar to lift, if items are independent, the conviction is 1.



frequentItemSets = apriori(basketsets, min_support = 0.02, use_colnames = True)
myrules = association_rules(frequentItemSets, metric = 'lift', min_threshold = 1)
myrules = myrules.sort_values(by=['confidence'], ascending = False)
myrules
# If you were running this outside of jupyter uncomment below to print out results
# print(myrules.head(100))


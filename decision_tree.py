import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix



df = pd.read_csv('./input/processed.cleveland.data')

def print_pd_head(num):
    return print(df.head(num))
# print(df.head())

def print_df_dtypes(df):
    return print(df.dtypes)

df.columns = [
    'age',
    'sex',
    'cp',
    'restbp',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal',
    'hd'
]

"""
looking at the unique values in the ca and thal columns

"""
# print(df['ca'].unique())
# print(df['thal'].unique())

"""
dealing with the missing data
"""
# print(len(df.loc[(df['ca'] == '?')
# |
# (df['thal'] == '?')]))
# print(df.loc[(df['ca'] == '?')
# |
# (df['thal'] == '?')])

# print(len(df))

"""
dropping the rows with missing values by creating a new dataset that has everthing as before but does not contain the rows with missing values(since the missing values only occupied 3 rows)
"""
df_no_missing = df.loc[(df['ca'] != '?')
&
(df['thal'] != '?')]

"""

verifying that weve donw this

"""

# print(df_no_missing)
# print(df_no_missing['ca'].unique())
# print(df_no_missing['thal'].unique())


"""
formatting the data for our tree
"""

X = df_no_missing.drop('hd', axis=1).copy()
# print(X.head())
y = df_no_missing['hd'].copy()
# print(y)

"""
Onehot encoding
(in order to use categorical data in sklearn we need to onehot encode)
we will be using pandas.getdummies, we can also use columntransformer
"""
# print(X['cp'].unique())

# print(pd.get_dummies(X, columns=['cp']).head())
X_encoded = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'])

# print(X_encoded.head())

# print(y.unique())

y_not_zero_index = y > 0 # get the index for each non-zero value in y
y[y_not_zero_index] = 1
# print(y.unique())



"""
building the tree
"""
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

# print(clf_dt.predict_proba(X_test, check_input=True))

"""

cross complexitt pruning

"""

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]


clf_dts = [] #creating an array of decision trees

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)



"""
using cross validation
"""

clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016)
scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
# print(scores)


alpha_loop_values = []

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])
# print(alpha_results)

ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014)
&
(alpha_results['alpha'] < 0.015)
]

#convert ideal_ccp_alpha from series to a float
# ideal_ccp_alpha = float(ideal_ccp_alpha)





"""
building the final classificaiton tree
"""
clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=0.014)

clf_dt_pruned.fit(X_train, y_train)

print(clf_dt_pruned.score(X_test, y_test))
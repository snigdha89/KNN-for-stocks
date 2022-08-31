from sklearn.model_selection import train_test_split
from sklearn . preprocessing import StandardScaler
from sklearn . metrics import confusion_matrix
from sklearn . neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
pd.options.mode.chained_assignment = None

goog_path = os.path.abspath('GOOG_weekly_return_volatility.csv')
df_goog = pd.read_csv(goog_path)
df_googvol = df_goog[df_goog.Year.isin([2019])]
df_googvol_2yrs = df_goog[df_goog.Year.isin([2019,2020])]

print('##############Q1################')
error_rate = []
Pred_lst  =[]
Y_test_lst = []
k_lst = [3,5,7,9,11]

for k in range (3 ,13 ,2):
    X = df_googvol [["mean_return", "volatility"]]
    y = df_googvol["Label"]
    scaler = StandardScaler (). fit (X)
    X = scaler . transform (X)
    X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5)
    knn_classifier = KNeighborsClassifier ( n_neighbors =k)
    knn_classifier.fit(X_train , Y_train )
    pred_k = knn_classifier.predict(X_test)
    Pred_lst.append(pred_k)
    error_rate.append(np.mean(pred_k == Y_test))
    Y_test_lst.append(Y_test)
    print("The Accuracy is {} when k is {} for 2019 ".format(np.mean(pred_k == Y_test), k )) 
print("All Accuracies for 2019 are  : ", error_rate)
plt.figure ( figsize =(10 ,4))
ax = plt. gca ()
ax.xaxis.set_major_locator ( plt.MaxNLocator ( integer = True ))
plt.plot ( range (3 ,13 ,2) , error_rate , color ='red', linestyle ='dotted',
marker ='o', markerfacecolor ='black', markersize =10)
plt.title ('Accuracy vs. k ')
plt.xlabel ('number of neighbors : k')
plt.ylabel ('Accuracy')


max_value = max(error_rate)
max_index = error_rate.index(max_value)
print('Optimal value for k in 2019 is : ', k_lst[max_index])

## Q1. Part 2 - Predicting labels for 2020 using optimal k from 2019 

df_googvol_2yrs_test = pd.DataFrame (
{"mean_return": df_googvol_2yrs.iloc[:,2].tolist(),
"volatility":df_googvol_2yrs.iloc[:,3].tolist(),
"Label":df_googvol_2yrs.iloc[:,4].tolist()},
columns = ["mean_return","volatility", "Label"])
X = df_googvol_2yrs_test [["mean_return", "volatility"]]
y = df_googvol_2yrs_test["Label"]
scaler = StandardScaler (). fit (X)
X = scaler . transform (X)
X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5, shuffle = False)
knn_classifier = KNeighborsClassifier ( n_neighbors = k_lst[max_index])
knn_classifier.fit(X_train , Y_train )
pred_k_2020 = knn_classifier.predict(X_test)
print("pred_k_2020", pred_k_2020)
error_rate.append(np.mean(pred_k_2020 == Y_test))
print("The Accuracy is {} when k is {}  for year 2020 ".format(np.mean(pred_k_2020 == Y_test), k_lst[max_index] )) 


###Q1( part 3,4)  -> Confusion matrix , TNR and TPR for year 2 ie 2020 ####
cf_1 = confusion_matrix( Y_test , pred_k_2020 )
print("Confusion matrix for year 2020  for k {} is {} ".format(k_lst[max_index], cf_1))
tpr = cf_1[1][1]/(cf_1[1][1] + cf_1[1][0])
tnr = cf_1[0][0]/(cf_1[0][0] + cf_1[0][1])
print("TPR  for year 2020 is {}  and TNR for year 2020 is {}".format( tpr, tnr))

print('################# Labels buy and hold and trading Strategy ###########')

googd_path = os.path.abspath('GOOG_weekly_return_volatility_detailed.csv')

df_googvold = pd.read_csv(googd_path,parse_dates=["Date"],dayfirst=True).drop(['High','Low','Open','Close','Volume', 'mean_return','volatility'], axis=1).sort_values('Date')
df_googvold['Open'] = df_googvold['Adj Close'].shift(1)
df_googvold['Close'] = df_googvold['Adj Close']
df_googvold = df_googvold.drop(['Adj Close'], axis = 1)

df_googvold = df_googvold[df_googvold.Year.isin([2020])]
df_goog = df_googvold.groupby(['Year','Week_Number']).agg({'Date': ['min','max']}).reset_index()
df_goog.columns = ['Year','Week_Number','OpenDate','CloseDate']
df_goog = (df_goog.merge(df_googvold[['Date','Open']], left_on = 'OpenDate', right_on = 'Date')
      .drop('Date',axis=1)
      .merge(df_googvold[['Date','Close']], left_on = 'CloseDate', right_on = 'Date')
      .drop('Date',axis=1))

df_goog = df_goog.merge(df_googvol_2yrs[['Week_Number','Year','Label']],how='left',left_on=['Week_Number','Year'],right_on = ['Week_Number','Year'])
df_goog['NexLabel'] = df_goog['Label'].shift(-1)


cap = 100 + 100*(df_goog.loc[52,'Close'] - df_goog.loc[0,'Open'])/df_goog.loc[0,'Open']
print("GOOG buy-hold  cap for 2020 : {}".format(cap))

cap  = 100
op = 0
for index, row in df_goog.iterrows():
    if row[6] == 1 and op == 0:
        op = row[4]
    if row[6] == 1 and row[7] == 0:
        cap = cap + cap * ((row[5] - op)/op)
        op = 0

print("GOOG trading startegy based on label cap for 2020 : {}".format(cap))
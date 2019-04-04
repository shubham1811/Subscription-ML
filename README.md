Code Explanation:-

###fintech_app.py###

---------------------------------------
IMPORTED THE LIBRARY
---------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from dateutil import parser


--------------------------------------
Read the File
--------------------------------------

dataset = pd.read_csv('appdata10.csv')

-----------------------------------------------
Sliced the Hour Dataset and converted into int
-----------------------------------------------

dataset['hour'] = dataset.hour.str.slice(1,3).astype(int)

--------------------------------------------------------
Copied the dataset into dataset2 and droped the column
--------------------------------------------------------

dataset2 = dataset.copy().drop(columns=['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])


---------------------------------------------------------
Drawn The Graph of each column
---------------------------------------------------------
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i - 1])
    vals = np.size(dataset2.iloc[:, i - 1].unique())
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

[Graph1](https://drive.google.com/open?id=1JO3-C7fsSwJ_FDM2bnsd8bseQZQtHqeB)
---------------------------------------------------------
Graph how enrollled subscription depend upon no of column
---------------------------------------------------------

dataset2.corrwith(dataset.enrolled).plot.bar(figsize = (20,10), title ='Correlation with Response Variable',fontsize = 15, rot= 45, grid =True

[Graph2](https://drive.google.com/open?id=118ogIKaIcWD_YTvuDpyTtUyEjlj4m8vb)

----------------------------------------------------------
Correlation matrix of each column
----------------------------------------------------------
sn.set(style="white", font_scale=2)

corr=dataset2.corr()

mask=np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)

cmap = sn.diverging_palette(220, 10, as_cmap =True)

sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

[Graph3](https://drive.google.com/open?id=1SCktEO_mChf7R-4a8xoflCVqB7P4dSFC)


----------------------------------------------------------------------
Parsing the Enrolled and first open column
----------------------------------------------------------------------

dataset["first_open"] = [parser.parse(row_date) for row_date in dataset["first_open"]]
dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in dataset["enrolled_date"]]


--------------------------------------------------------------------------------------------------
Taken the difference of enrolled_date-first_open and added the difference column in dataset
--------------------------------------------------------------------------------------------------
dataset["difference"] = (dataset.enrolled_date-dataset.first_open).astype('timedelta64[h]')

---------------------------------------------------------------------------------------
Drawn the graph how many enrollment take place in hour
---------------------------------------------------------------------------------------

plt.hist(dataset["difference"].dropna(), color='#3F5D7D')
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()

[Graph4](https://drive.google.com/open?id=1sgxvmKp8TPyTCk0qMRbh6ZxJ5SmKWRzL)

---------------------------------------------------------------------------------------------------------
Manipulated the value of the enrolled column and droped the difference,enrolled_date,first_open column
---------------------------------------------------------------------------------------------------------

dataset.loc[dataset.difference > 48, 'enrolled'] = 0
dataset = dataset.drop(columns = ['difference', 'enrolled_date', 'first_open'])



---------------------------------------------------------------------------------------------------------
Taken the 2nd Dataset
---------------------------------------------------------------------------------------------------------

top_screens=pd.read_csv('top_screens.csv').top_screens.values


---------------------------------------------------------------------------------------------------------
Converted the value of screen_list column in string
---------------------------------------------------------------------------------------------------------

dataset["screen_list"] = dataset.screen_list.astype(str)+','

---------------------------------------------------------------------------------------------------------
Created the column in dataset that are in screen_list datatype
----------------------------------------------------------------------------------------------------------

for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset["screen_list"] = dataset.screen_list.str.replace(sc+",", "")

---------------------------------------------------------------------------------------------------------
Counted the string left in screen_list column and droped the screen_list column
----------------------------------------------------------------------------------------------------------

dataset["other"] = dataset.screen_list.str.count(",")
dataset =dataset.drop(columns=["screen_list"])


---------------------------------------------------------------------------------------------------------
Added the Screens and pasted it in the dataset
----------------------------------------------------------------------------------------------------------

savings_screens = ["Saving1",
                   "Saving2",
                   "Saving2Amount",
                   "Saving4",
                   "Saving5",
                   "Saving6",
                   "Saving7",
                   "Saving8",
                   "Saving9",
                   "Saving10",
                   ]

#Added the Saving_Screens and pasted it in the dataset["SavingCount"]
dataset["SavingCount"] = dataset[savings_screens].sum(axis=1)

dataset = dataset.drop(columns=savings_screens)


cm_screens =["Credit1",
             "Credit2",
             "Credit3",
             "Credit3Container",
             "Credit3Dashboard",
             ]

dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)



cc_screens= ["CC1",
             "CC1Category",
             "CC3"]

dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)

loan_screens = ["Loan",
                "Loan2",
                "Loan3",
                "Loan4"]

dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)




### Modal.py ####

---------------------------------------------------------------------------------------------------------
Created the new dataset
---------------------------------------------------------------------------------------------------------

dataset.to_csv("new_appdata10.csv", index = False)

---------------------------------------------------------------------------------------------------------
Imported the data set
---------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time

---------------------------------------------------------------------------------------------------------
Read the data set
---------------------------------------------------------------------------------------------------------

dataset = pd.read_csv('new_appdata10.csv')

---------------------------------------------------------------------------------------------------------
Taken the enrolled column value in response and droped the enrolled column
---------------------------------------------------------------------------------------------------------

response = dataset["enrolled"]
dataset = dataset.drop(columns = 'enrolled')

---------------------------------------------------------------------------------------------------------
Spit the dataset into training set and test set
---------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataset, response,test_size = 0.2,random_state = 0)



---------------------------------------------------------------------------------------------------------
Taken the user value in another variable and droped user column 
---------------------------------------------------------------------------------------------------------

train_identifier= X_train['user']
X_train = X_train.drop(columns= 'user')
test_identifier = X_test['user']
X_test = X_test.drop(columns = 'user')


---------------------------------------------------------------------------------------------------------
Feature Scaling of train and test data 
---------------------------------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2


---------------------------------------------------------------------------------------------------------
Trained the training data using LogisticRegression
---------------------------------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,penalty = 'l1')
classifier.fit(X_train, Y_train)

---------------------------------------------------------------------------------------------------------
Predicted the data by giving argument as X_test
---------------------------------------------------------------------------------------------------------

y_pred = classifier.predict(X_test)

---------------------------------------------------------------------------------------------------------
Confusion Matrix
---------------------------------------------------------------------------------------------------------

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm=confusion_matrix(Y_test, y_pred)

---------------------------------------------------------------------------------------------------------
Accuracy Score
---------------------------------------------------------------------------------------------------------

accuracy_score(Y_test, y_pred)

[Accuracy](https://drive.google.com/open?id=1k5OrsMWkvWTqO1JY96h-KOwWS0N7sGr-)

---------------------------------------------------------------------------------------------------------
Confusion Matrix
---------------------------------------------------------------------------------------------------------

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(Y_test, y_pred))


[Confusion matix](https://drive.google.com/open?id=1FkNk9BqERBpj-F-ZN9rS_TyhJnxKFDL6)






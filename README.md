# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income.csv",na_values=[ " ?"])
data
```

<img width="1332" height="665" alt="image" src="https://github.com/user-attachments/assets/b2fb31c8-a020-41f7-b0cd-5b71f21a6edf" />


```
data.isnull().sum()
```
<img width="453" height="682" alt="image" src="https://github.com/user-attachments/assets/0b0610ec-7428-4607-b6f4-d23ed0c825ce" />


```
missing=data[data.isnull().any(axis=1)]
missing
```
<img width="1417" height="717" alt="image" src="https://github.com/user-attachments/assets/6dfa105f-a7a9-4574-8975-8f05144b6e2e" />


```
data2=data.dropna(axis=0)
data2
```
<img width="1392" height="674" alt="image" src="https://github.com/user-attachments/assets/4aa7737e-2d44-432a-9d68-ebf748288b14" />


```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
<img width="1106" height="507" alt="image" src="https://github.com/user-attachments/assets/641b59f9-f66e-42e7-a7cd-01d8b3bf5d75" />


```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="567" height="682" alt="image" src="https://github.com/user-attachments/assets/9c6c2034-243f-4867-a20b-630dedafe4b7" />


```
 data2
```

<img width="1366" height="712" alt="image" src="https://github.com/user-attachments/assets/b48d1f84-1e35-44a4-adc4-ce269c48892d" />


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1402" height="691" alt="image" src="https://github.com/user-attachments/assets/1c338583-ea18-4b7b-b08a-cc748a012464" />


```
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1426" height="135" alt="image" src="https://github.com/user-attachments/assets/8249edc4-7a0f-454b-8202-aeac33ba00ca" />


```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="1400" height="141" alt="image" src="https://github.com/user-attachments/assets/f7ee5b18-7e1e-4611-bd43-65bf31bdfd2a" />


```       
y=new_data['SalStat'].values
print(y)
```

<img width="351" height="116" alt="image" src="https://github.com/user-attachments/assets/2e24b341-b262-4e1b-8240-3de790405702" />


```
x=new_data[features].values
print(x)
```
<img width="505" height="251" alt="image" src="https://github.com/user-attachments/assets/a766f7f7-9046-4dcf-9ae4-f63f9565cfc1" />


```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

<img width="873" height="197" alt="image" src="https://github.com/user-attachments/assets/ab542c3f-9708-4b76-a050-e2e43d4a4daa" />


```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="566" height="170" alt="image" src="https://github.com/user-attachments/assets/1f2f5cd9-291d-4a77-8a5e-d1e5d2296121" />


```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
<img width="532" height="146" alt="image" src="https://github.com/user-attachments/assets/47cf975d-a101-4bb3-80ef-868e9fef7932" />



```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="712" height="145" alt="image" src="https://github.com/user-attachments/assets/c0987a43-1771-4a39-840e-8933460f9235" />



```
data.shape
```
<img width="221" height="100" alt="image" src="https://github.com/user-attachments/assets/497a8cba-158b-457b-be55-edb0735ebca3" />


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="1041" height="556" alt="image" src="https://github.com/user-attachments/assets/5953e5c7-76c9-4039-b20e-1799ee0884cd" />


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="647" height="485" alt="image" src="https://github.com/user-attachments/assets/912a418b-d2cc-464a-9828-ed57d6e70adb" />


```
tips.time.unique()
```

<img width="572" height="147" alt="image" src="https://github.com/user-attachments/assets/98de30fe-8949-44fc-a2ff-4e383c49704e" />


```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="617" height="190" alt="image" src="https://github.com/user-attachments/assets/d2f5e17e-f1c3-4342-9128-2c15ee0d4326" />


```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
<img width="527" height="175" alt="image" src="https://github.com/user-attachments/assets/aaeecb8c-77f9-450d-9ec2-26fdd84116f6" />


```


```

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.



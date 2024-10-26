import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

test_data = pd.read_csv('test-data.csv')
train_data = pd.read_csv('train-data.csv')

train_data.head()

degisken_turu = train_data.copy()
print(degisken_turu.dtypes)

print(degisken_turu.shape)

nan_iceren = [col for col in degisken_turu.columns if degisken_turu[col].isnull().any()]
print(nan_iceren)

degisken_turu.describe()

sayisal=[col for col in degisken_turu.columns if degisken_turu[col].dtypes !='O']
kategori=[col for col in degisken_turu.columns if col not in sayisal]
print((kategori), "\n")
print(sayisal)

plt.figure(figsize=(10,6))
sns.heatmap(degisken_turu[sayisal].corr().abs(),annot=True)


for col in sayisal:
    sns.regplot(degisken_turu[col],degisken_turu['Price'])
    plt.show()


degisken_turu = degisken_turu.drop('New_Price',axis=1)


sns.set_style("darkgrid")
for col in ['Transmission','Fuel_Type','Location','Owner_Type','Engine']:
    degisken_turu.groupby(col).Price.mean().plot.bar()
    plt.title(col)
    plt.show()

    
train_dt=degisken_turu.copy()
train_dt.head()                                    
test_data.head()

test_dt = test_data.copy()
test_dt = test_dt.drop('New_Price',axis=1)


print(test_data.shape)

plt.figure()
sns.heatmap(test_data.isnull(),cbar=False,yticklabels=False)


son_dt=pd.concat([train_dt,test_dt],axis=0)
son_dt.head()

X=son_dt.iloc[:,:-1]
X_train=X.iloc[:len(train_dt),:]
X_test=X.iloc[len(train_dt):,:]


train_data=pd.concat([X_train,train_dt['Price']],axis=1)
train_data.head()


y_train=train_dt['Price'] 
y_train_log=np.log(y_train)
sns.distplot(y_train_log,kde=False)

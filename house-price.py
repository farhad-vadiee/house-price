import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression

# import requests
# file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'

# r = requests.get(file_name, allow_redirects=True)
# open('kc_house_data_NaN.csv', 'wb').write(r.content)

file_name="kc_house_data_NaN.csv"
df = pd.read_csv(file_name)
print(df)


print(df.dtypes)

df.drop(columns=["id","Unnamed: 0"], inplace=True)

print(df.describe())

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

floor_counts = df['floors'].value_counts()
floor_counts_df = floor_counts.to_frame()
print(floor_counts)

sns.boxplot(x='waterfront', y='price', data=df)

plt.title('Price Distribution by Waterfront View')
plt.xlabel('Waterfront View (0 = No, 1 = Yes)')
plt.ylabel('Price')
plt.show()

sns.regplot(x='sqft_above', y='price', data=df)

plt.title('Relationship between Sqft Above and Price')
plt.xlabel('Sqft Above')
plt.ylabel('Price')
plt.show()

# We can use the Pandas method corr() to find the feature other than price that is most correlated with price.
# df.corr()['price'].sort_values()

# We can Fit a linear regression model using the longitude feature 'long' and caculate the R^2.
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
print("long and price linear fit:",lm.score(X, Y))

X1 = df[['sqft_living']]
Y1 = df['price']
lm1 = LinearRegression()
lm1.fit(X1,Y1)

print("sqft_living and price linear fit:",lm1.score(X1, Y1))

features = ["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]   
X3 = df[features]
lm2 = LinearRegression()
lm2.fit(X3,Y)
err = lm2.score(X3,Y)
print(err)

Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
Z = df[features].astype(float)
pipe.fit(Z,Y)
pipe.score(Z,Y)
print(pipe.score(Z,Y))

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])
from sklearn.linear_model import Ridge

RigeModel=Ridge(alpha=0.1)
RigeModel.fit(x_train, y_train)
RigeModel.score(x_test, y_test)

# print(RigeModel.score(y_test,yhat))

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
polyRidge = Ridge(alpha=0.1)
polyRidge.fit(x_train_pr,y_train)
print(polyRidge.score(x_test_pr,y_test))

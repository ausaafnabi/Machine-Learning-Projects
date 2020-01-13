from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import os


DOWNLOAD_ROOT = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv"
DATASET_PATH = os.path.join("../","datasets")
DATASET_URL = DOWNLOAD_ROOT 

csv_file_path = os.path.join(DATASET_PATH,"FuelConsumption.csv")
df = pd.read_csv(csv_file_path)
df.head()
df.describe()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='blue')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

##
## MULTIPLE LINEAR REGRESSION
##

from sklearn import linear_model

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit(x,y)
# the coefficients
print('Coefficients: ',regr.coef_)
print('Intercept: ',regr.intercept_)
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color = 'blue')
plt.plot(x,regr.coef_[0][0]*x + regr.intercept_[0], '-r')
plt.xlabel('ENGINESIZE')
plt.ylabel('Emission')
plt.show()

#print('Coefficients: ',regr.coef_)

y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of the squares: %.2f" % np.mean((y_hat -y)**2))

# Explained variance score: 1 is perfect prediction

print ('Variance score: %.2f' %regr.score(x,y))


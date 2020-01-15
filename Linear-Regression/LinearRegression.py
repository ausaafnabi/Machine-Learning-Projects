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

# Histogram of these labels
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
cdf.hist()
plt.show()

#Comparative graph of the comsumption and CO2EMMISION
plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS)
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

#comprative graph of the ENGINESIZE and CO2EMISSON
plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='red')
plt.xlabel('ENGINESIZE')
plt.ylabel('Emission')
plt.show()

#comprative graph of the ENGINESIZE and CO2EMISSON
plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color='red')
plt.xlabel('Cylinder')
plt.ylabel('Emission')
plt.show()

##
## USING THE DATASETS DETAIL TO CREATE SIMPLE OF REGRESSION
##

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
# the Coefficients
print('Coefficients: ',regr.coef_)
print('Intercept: ',regr.intercept_)
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color = 'blue')
plt.plot(train_x,regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel('ENGINESIZE')
plt.ylabel('Emission')
plt.show()

##
## CASE ANALYSIS: ERROR ANALYSIS
##

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print ("Mean Absolute Error: %.2f" %np.mean(np.absolute(test_y_hat - test_y)))
print ("Mean quare Error (MSE) : %.2f" %np.mean((test_y_hat - test_y)**2))
print("R2-score: %.2f" %r2_score(test_y_hat, test_y))
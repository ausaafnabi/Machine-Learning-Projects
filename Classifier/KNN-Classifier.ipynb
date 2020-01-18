{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# KNN-CLASSIFIER for teleCustomer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h4>The K- Nearest Neighbour Algorithm to classify the peoples on the basis of services.</h4>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>region</th>\n",
       "      <th>tenure</th>\n",
       "      <th>age</th>\n",
       "      <th>marital</th>\n",
       "      <th>address</th>\n",
       "      <th>income</th>\n",
       "      <th>ed</th>\n",
       "      <th>employ</th>\n",
       "      <th>retire</th>\n",
       "      <th>gender</th>\n",
       "      <th>reside</th>\n",
       "      <th>custcat</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>13</td>\n",
       "      <td>44</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>64.0</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>136.0</td>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>68</td>\n",
       "      <td>52</td>\n",
       "      <td>1</td>\n",
       "      <td>24</td>\n",
       "      <td>116.0</td>\n",
       "      <td>1</td>\n",
       "      <td>29</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>33</td>\n",
       "      <td>33</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>33.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "      <td>23</td>\n",
       "      <td>30</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>30.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   region  tenure  age  marital  address  income  ed  employ  retire  gender  \\\n",
       "0       2      13   44        1        9    64.0   4       5     0.0       0   \n",
       "1       3      11   33        1        7   136.0   5       5     0.0       0   \n",
       "2       3      68   52        1       24   116.0   1      29     0.0       1   \n",
       "3       2      33   33        0       12    33.0   2       0     0.0       1   \n",
       "4       2      23   30        1        9    30.0   1       2     0.0       0   \n",
       "\n",
       "   reside  custcat  \n",
       "0       2        1  \n",
       "1       6        4  \n",
       "2       2        3  \n",
       "3       1        1  \n",
       "4       4        3  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import itertools\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.ticker import NullFormatter\n",
    "import pandas as pd\n",
    "import matplotlib.ticker as ticker\n",
    "from sklearn import preprocessing\n",
    "import os\n",
    "%matplotlib inline\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>region</th>\n",
       "      <th>tenure</th>\n",
       "      <th>age</th>\n",
       "      <th>marital</th>\n",
       "      <th>address</th>\n",
       "      <th>income</th>\n",
       "      <th>ed</th>\n",
       "      <th>employ</th>\n",
       "      <th>retire</th>\n",
       "      <th>gender</th>\n",
       "      <th>reside</th>\n",
       "      <th>custcat</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>13</td>\n",
       "      <td>44</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>64.0</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>136.0</td>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>68</td>\n",
       "      <td>52</td>\n",
       "      <td>1</td>\n",
       "      <td>24</td>\n",
       "      <td>116.0</td>\n",
       "      <td>1</td>\n",
       "      <td>29</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>33</td>\n",
       "      <td>33</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>33.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "      <td>23</td>\n",
       "      <td>30</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>30.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   region  tenure  age  marital  address  income  ed  employ  retire  gender  \\\n",
       "0       2      13   44        1        9    64.0   4       5     0.0       0   \n",
       "1       3      11   33        1        7   136.0   5       5     0.0       0   \n",
       "2       3      68   52        1       24   116.0   1      29     0.0       1   \n",
       "3       2      33   33        0       12    33.0   2       0     0.0       1   \n",
       "4       2      23   30        1        9    30.0   1       2     0.0       0   \n",
       "\n",
       "   reside  custcat  \n",
       "0       2        1  \n",
       "1       6        4  \n",
       "2       2        3  \n",
       "3       1        1  \n",
       "4       4        3  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "DATASET_PATH = os.path.join(\"../\",\"datasets\")\n",
    "\n",
    "csv_file_path1 = os.path.join(DATASET_PATH,\"teleCust1000t.csv\")\n",
    "df = pd.read_csv(csv_file_path,error_bad_lines=False)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3    281\n",
       "1    266\n",
       "4    236\n",
       "2    217\n",
       "Name: custcat, dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['custcat'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f12d31e0668>]],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.hist(column = 'income',bins=50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',\n",
       "       'employ', 'retire', 'gender', 'reside', 'custcat'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  2.,  13.,  44.,   1.,   9.,  64.,   4.,   5.,   0.,   0.,   2.],\n",
       "       [  3.,  11.,  33.,   1.,   7., 136.,   5.,   5.,   0.,   0.,   6.],\n",
       "       [  3.,  68.,  52.,   1.,  24., 116.,   1.,  29.,   0.,   1.,   2.],\n",
       "       [  2.,  33.,  33.,   0.,  12.,  33.,   2.,   0.,   0.,   1.,   1.],\n",
       "       [  2.,  23.,  30.,   1.,   9.,  30.,   1.,   2.,   0.,   0.,   4.]])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = df[['region','tenure', 'age', 'marital', 'address', 'income', 'ed',\n",
    "       'employ', 'retire', 'gender', 'reside']].values\n",
    "X[0:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 4, 3, 1, 3])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = df['custcat'].values\n",
    "y[0:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.02696767, -1.055125  ,  0.18450456,  1.0100505 , -0.25303431,\n",
       "        -0.12650641,  1.0877526 , -0.5941226 , -0.22207644, -1.03459817,\n",
       "        -0.23065004],\n",
       "       [ 1.19883553, -1.14880563, -0.69181243,  1.0100505 , -0.4514148 ,\n",
       "         0.54644972,  1.9062271 , -0.5941226 , -0.22207644, -1.03459817,\n",
       "         2.55666158],\n",
       "       [ 1.19883553,  1.52109247,  0.82182601,  1.0100505 ,  1.23481934,\n",
       "         0.35951747, -1.36767088,  1.78752803, -0.22207644,  0.96655883,\n",
       "        -0.23065004],\n",
       "       [-0.02696767, -0.11831864, -0.69181243, -0.9900495 ,  0.04453642,\n",
       "        -0.41625141, -0.54919639, -1.09029981, -0.22207644,  0.96655883,\n",
       "        -0.92747794],\n",
       "       [-0.02696767, -0.58672182, -0.93080797,  1.0100505 , -0.25303431,\n",
       "        -0.44429125, -1.36767088, -0.89182893, -0.22207644, -1.03459817,\n",
       "         1.16300577]])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))\n",
    "X[0:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train Set :  (800, 11) (800,)\n",
      "Test Set :  (200, 11) (200,)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)\n",
    "print('Train Set : ', X_train.shape,y_train.shape)\n",
    "print('Test Set : ', X_test.shape,y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### CLASSIFICATION::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier as KNC\n",
    "k = 38\n",
    "neighbor = KNC(n_neighbors = k).fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3, 1, 2, 4, 4])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "yhat = neighbor.predict(X_test)\n",
    "yhat[0:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test accuracy ::  0.4275\n",
      "Test accuracy ::  0.41\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "print('Train accuracy :: ', metrics.accuracy_score(y_train,neighbor.predict(X_train)))\n",
    "print('Test accuracy :: ', metrics.accuracy_score(y_test,yhat))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.3  , 0.29 , 0.315, 0.32 , 0.315, 0.31 , 0.335, 0.325, 0.34 ,\n",
       "       0.33 , 0.315, 0.34 , 0.33 , 0.315, 0.34 , 0.36 , 0.355, 0.35 ,\n",
       "       0.345, 0.335, 0.35 , 0.36 , 0.37 , 0.365, 0.365, 0.365, 0.35 ,\n",
       "       0.36 , 0.38 , 0.385, 0.395, 0.395, 0.38 , 0.37 , 0.365, 0.385,\n",
       "       0.395, 0.41 , 0.395, 0.395, 0.395, 0.38 , 0.39 , 0.375, 0.365,\n",
       "       0.38 , 0.375, 0.375, 0.365, 0.36 , 0.36 , 0.365, 0.37 , 0.38 ,\n",
       "       0.37 , 0.37 , 0.37 , 0.36 , 0.35 , 0.36 , 0.355, 0.36 , 0.36 ,\n",
       "       0.36 , 0.34 , 0.34 , 0.345, 0.35 , 0.35 , 0.355, 0.365, 0.355,\n",
       "       0.355, 0.365, 0.37 , 0.37 , 0.37 , 0.35 , 0.35 , 0.35 , 0.35 ,\n",
       "       0.36 , 0.355, 0.33 , 0.32 , 0.345, 0.345, 0.345, 0.335, 0.345,\n",
       "       0.355, 0.345, 0.345, 0.34 , 0.34 , 0.335, 0.345, 0.325, 0.315])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Ks = 100\n",
    "mean_acc = np.zeros((Ks-1))\n",
    "std_acc = np.zeros((Ks-1))\n",
    "ConfustionMx = [];\n",
    "for n in range(1,Ks):\n",
    "    \n",
    "    #Train Model and Predict  \n",
    "    neighbor = KNC(n_neighbors = n).fit(X_train,y_train)\n",
    "    yhat=neighbor.predict(X_test)\n",
    "    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)\n",
    "\n",
    "    \n",
    "    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])\n",
    "\n",
    "mean_acc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(range(1,Ks),mean_acc,'g')\n",
    "plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)\n",
    "plt.legend(('Accuracy ', '+/- 3xstd'))\n",
    "plt.ylabel('Accuracy ')\n",
    "plt.xlabel('Number of Nabors (K)')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best accuracy was with 0.41 with k= 38\n"
     ]
    }
   ],
   "source": [
    "print( \"The best accuracy was with\", mean_acc.max(), \"with k=\", mean_acc.argmax()+1) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

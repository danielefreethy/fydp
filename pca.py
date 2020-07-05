import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns; sns.set(style='white')
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp
import pandas as pd


def anomaly_scores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF)-np.array(reducedDF))**2, axis=1)
    loss = pd.Series(data=loss,index=originalDF.index)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
    return loss


# Returns preds which is a df with two columns (each point has its anomaly_score vs its actual Fault status)
def predict(trueLabels, anomalyScores):
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']
    return preds


# Loading the dataset
data = pd.read_csv("Good-Batch-Data.csv")
data = data.drop(columns=['X_offline', 'Viscosity_offline', 'P_offline', 'NH3_offline', 'PAA_offline'])

# Split into data and target sets (Result is our target feature)
dataX = data.copy().drop(['Fault_ref'], axis=1)
dataY = data['Fault_ref'].copy()

# Normalize Data
featuresToScale = dataX.columns
sX = pp.StandardScaler(copy=True)
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=2018, stratify=dataY)

# PCA Stuff (Not sure if I did this right)
n_components = 30
whiten = False
random_state = 2018

pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)

# Not sure why we do an inverse transform after PCA...
X_train_PCA_inverse = pca.inverse_transform(X_train_PCA)
X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse, index=X_train.index)

# Trying to compare
# Use their anomaly function to compare original dataset w PCA set
anomalyScoresPCA = anomaly_scores(X_train, X_train_PCA_inverse)
preds = predict(y_train, anomalyScoresPCA)

# Takes the 350 most anomalous data points
preds.sort_values(by="anomalyScore", ascending=False, inplace=True)
cutoff = 350
predsTop = preds[:cutoff]
print(predsTop)

# Precision is how many it gets right from the top 350, then recall is how faults it finds from whole set?
print("Precision: ", predsTop.anomalyScore[predsTop.trueLabel == 0].count()/cutoff)
recall = 1.0
if y_train.sum() > 0:
    recall = predsTop.anomalyScore[predsTop.trueLabel == 1].count()/y_train.sum()
print("Recall: ", recall)

# This data only has valid examples, so precision will be 100% then recall will be 350/size of whole set

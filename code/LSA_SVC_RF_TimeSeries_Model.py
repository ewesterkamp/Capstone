from Master_Data import MasterData
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def vectorizeData(self,data,showinfo=True):

    X_train = data['data']

    # Vectorize the data and look at the key words in the corpus
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
        
    X = vectorizer.fit_transform(X_train)

    names = vectorizer.get_feature_names()
    vocab = vectorizer.vocabulary_
    weights = np.asarray(X.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': names, 'weight': weights})
    sortedVals = weights_df.sort_values(by='weight', ascending=False).head(10)
    if(showinfo):
        print(sortedVals)
        print("Number of documents/headers in selection is {}".format(X.shape[0]))
        print("Number of unique words in Training Set is {}".format(X.shape[1]))

        if(self.showPlots):
            ax = sortedVals.plot(kind='bar', title='Bloomberg Index Top Words', figsize=(10,6), width=.8, fontsize=14, rot=45,x='term' )
            ax.title.set_size(20)
            plt.show()

    return X,vectorizer


# Take a numpy array for features and an equal length array of labels and return 
# a 3D array for the X that is based on Time Series.  Also return labels that has been
# adjusted so that it matches the X array.  Return both a train and test set of data
#
# return X_train,y_train,X_test,y_test
# X_train and X_test is of shape [length][seq_len][features_len]
def makeTimeSeries(X,Y,seq_len=10,split=.8):
    result = []

    for index in range(len(X) - seq_len+1):        # maxmimum date = lastest date - sequence length
        result.append(X[index: index + seq_len]) # index : index + seqlen
            
    X_ts = np.array(result)
    Y_ts = Y[(seq_len)-1:]       # Note that the Y value for each window should be the value of the last day of the sequence. 

    row = round(split* X_ts.shape[0]) # 90% split
    X_train = X_ts[:int(row), :]    # 90% date
    Y_train = Y_ts[:int(row)]
    X_test = X_ts[int(row):,:]
    Y_test = Y_ts[int(row):]
    
    return X_train,Y_train,X_test,Y_test


# Choose LSA vs. Semantic based on system args
import sys
svc=False
if len(sys.argv)>0:
    for i in range(len(sys.argv)):
        if sys.argv[i]=="SVC":
             print("Running as SVC")
            svc=True
        else: 
            print("Runnind as RandomForest")

# Load the base data set
md = MasterData()
md.loadData()

naiveDF = md.getNaivePredictors()

print(naiveDF.head())

# Vectorize the data and look at the key words in the corpus
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
X = vectorizer.fit_transform(md.masterDF['data'])

#split = int(X.shape[0]*.8)

# Break out Train and test on the vectorized features
# X_train = X[:split]
# X_test = X[split:]

# Reduce the dimensions
svd = TruncatedSVD(n_components=450,random_state=42)
X_svd = svd.fit_transform(X)
# X_train_svd = X_svd[:split]
# X_test_svd = X_svd[split:]

# X_train_svd = svd.fit_transform(X_train)
# X_test_svd = svd.transform(X_test)
s = sum(svd.explained_variance_ratio_)

fname=""
if(svc):
    param_grid = [{'C': [1, 10, 100, 1000], 'gamma':    [0.01,0.001, 0.0001]}]
#    param_grid = [{'C': [10, 1000], 'gamma': [0.01,0.0001]}]
    clf = svm.SVC(kernel='rbf',class_weight='balanced')
    fname="LSA_SVC_"
else:
    param_grid=[{'n_estimators':[5,10,20,50],'min_samples_leaf':[1,2,5,10,20,50]}]
    clf = RandomForestClassifier(random_state=42,class_weight='balanced')
    fname="LSA_RandForest_"

seq_nums=[1,5,20]

metricInfo = {}
clfInfo={}
# For Each Index
for i in md.indexNames:
    for seq_len in seq_nums:
        metrics2 = {}

        #========== Run with TimeSeries TS =============
        # Get the data as a time-series
        # Create the train/test split on the data 
        Y = md.getLabelsForIndex(i)
        Y2 = np.array(Y["y2"])
        X_ts_train,Y_ts_train,X_ts_test,Y_ts_test = makeTimeSeries(X_svd,Y2,seq_len,.8)

        # flatten the X train and test data
        X_ts_train = X_ts_train.reshape(X_ts_train.shape[0],X_ts_train.shape[1]*X_ts_train.shape[2])
        X_ts_test = X_ts_test.reshape(X_ts_test.shape[0],X_ts_test.shape[1]*X_ts_test.shape[2])

        # Use the same fitted classifier from above 
        ### Run the Y2 predictions, only run GridSearchCV once
        grid = GridSearchCV(estimator=clf, param_grid=param_grid,scoring='f1')
        grid.fit(X_ts_train,Y_ts_train)
        pred = grid.predict(X_ts_test)

        # Write out the predict vs. test data for later analyis
        loc = -1*len(Y_ts_test)
        tmp = pd.DataFrame(md.masterDF.iloc[loc:][i+"_Adj Close"])
        tmp['y_test']=Y_ts_test
        tmp['pred']=pred

        pd.DataFrame(tmp).to_csv("../tmp/{}_{}_lsa_RF_predictions.csv".format(i,seq_len))

        clf2=grid.best_estimator_    

        tn, fp, fn, tp=confusion_matrix(Y_ts_test, pred).ravel()

        try:
            total_predictions = tn + fn + fp + tp
            accuracy = 1.0*(tp + tn)/total_predictions
            precision = 1.0*tp/(tp+fp)
            recall = 1.0*tp/(tp+fn)
            f1 = 2.0 * tp/(2*tp + fp+fn)
            f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
            metrics2.update({"Accuracy":accuracy})
            metrics2.update({"Precision":precision})
            metrics2.update({"Recall":recall})
            metrics2.update({"f1":f1})
            metrics2.update({"f2":f2})
            metrics2.update({"Total Predictions":total_predictions})
            metrics2.update({"True Pos":tp})
            metrics2.update({"True Neg":tn})
            metrics2.update({"False Pos":fp})
            metrics2.update({"False Neg":fn})
            metrics2.update({"CLF":clf2})
            print metrics2
        except:
            print "Got a divide by zero when trying out:", clf
            print "Precision or recall may be undefined due to a lack of true positive predicitons."

        metricInfo.update({i+" seq_len {}".format(seq_len):metrics2.copy()})
        

print("\n====Metrics ======")
print("\n")
print pd.DataFrame(metricInfo)
pd.DataFrame(metricInfo).to_csv("../data/"+fname+"metrics.csv")
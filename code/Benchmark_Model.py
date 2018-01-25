from Master_Data import MasterData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the base data set
md = MasterData()
md.loadData()

metricInfo = {}

# For each index figure out the metrics for a Naive Predictor
for i in md.indexNames:
    metrics = {}

    Y = md.getLabelsForIndex(i)
    Y2 = np.array(Y["y2"])

    # Split into just the test set size
    split = int(Y2.shape[0]*.8)

    y2_test = Y2[split:]
    pred = np.ones(len(y2_test))

    tn, fp, fn, tp=confusion_matrix(y2_test, pred).ravel()
    print(tn,fp,fn,tp)
    try:
        total_predictions = tn + fn + fp + tp
        accuracy = 1.0*(tp + tn)/total_predictions
        precision = 1.0*tp/(tp+fp)
        recall = 1.0*tp/(tp+fn)
        f1 = 2.0 * tp/(2*tp + fp+fn)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        metrics.update({"Accuracy":accuracy})
        metrics.update({"Precision":precision})
        metrics.update({"Recall":recall})
        metrics.update({"f1":f1})
        metrics.update({"f2":f2})
        metrics.update({"Total Predictions":total_predictions})
        metrics.update({"True Pos":tp})
        metrics.update({"True Neg":tn})
        metrics.update({"False Pos":fp})
        metrics.update({"False Neg":fn})

    except:
        print "Got a divide by zero when trying out:"
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

    metricInfo.update({i:metrics})  


print("\n====Naive Predictor 1 ======")
print("\n")
print pd.DataFrame(metricInfo)
pd.DataFrame(metricInfo).to_csv("../data/benchmark_naive_predictor.csv")

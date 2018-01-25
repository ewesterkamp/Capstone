
import datetime
import pandas as pd
import numpy as np
import glob
from textblob import TextBlob
from tqdm import tqdm
import pandas_datareader.data as web
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Master_Data import MasterData

stock_name="^GSPC"
indexNames = ['^GSPC','^DJI','MSCI']

def sorting(l1, l2):
    # l1 and l2 has to be numpy arrays
    idx = np.argsort(l1)
    return l1[idx], l2[idx]



class Evaluate_Data():

    indexData = {}
    haveIndexData=False
    polFactor = 0.25
    rawdata = pd.DataFrame()  # The data 

    startDate= datetime.datetime(2012,12,20)
    endDate = datetime.datetime(2017,12,1)

    saveFileName = "./data/{}_{}_feature_class_set.csv".format(stock_name,pd.to_datetime('today').strftime("%m%d%Y"))
    showPlots=True

    def __init__(self):
        pass

    def setDateRange(self,startDate,endDate):
        self.startDate=startDate
        self.endDateS= endDate

    # Loop through each principal component, sort out the top 10 words in it and display them
    def showComponents(self,names,svd,n_top=10):

        cnt=1
        for i,row in enumerate(svd.components_):
            print("==== Principal Component {} explained variance {} total features ====".format(cnt,svd.explained_variance_[i]))
            components = pd.DataFrame({'term': names, 'weight': row})
            sortedVals = components.sort_values(by='weight', ascending=False,axis=0).head(n_top)
            cnt+=1
            #print( sortedVals)


    # Given a list of indexes and a time-range, pull in key statistical information
    # Takes as input the MasterData data set and the indexNames
    def analyzeIndexes(self,md):


        analysis=[]
        plotDict = {}
        for indexName in md.indexNames:
            df = md.getIndexInfo(indexName)
            
            startOpen = df.iloc[0][indexName+'_Adj Close']
            endOpen = df.iloc[len(df)-1][indexName+'_Adj Close']
            valueChange = endOpen-startOpen
            growth = valueChange/startOpen*100
            cagr = (pow(endOpen/startOpen,(1.0/5.0))-1)
            avgTradVol = np.mean(df[indexName+'_Volume'])
            dict = {'index':indexName,'sOpen':startOpen,'eOpen':endOpen,
                    'valChange':valueChange,'growth':growth,
                    'cagr':cagr, 'tradingDays':len(df),
                    'avg trade vol':avgTradVol}
            analysis.append(dict)
            plotDict.update({indexName:df[indexName+'_Adj Close']})

        if(self.showPlots):
            pltDF = pd.DataFrame(plotDict)
            print( pltDF.head())
            pltDF.plot(secondary_y = ["^DJI"],grid=True,title="Index Value")    # Hardcoded here (arghh) to move DJI to secondary

            indx_return = pltDF.apply(lambda x: x / x[0])
            indx_return.plot(grid = True,title="Return since 1/1/2013").axhline(y = 1, color = "black", lw = 2)

            plt.show()


        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        for each in analysis:
            print("--- Index {} ---".format(each['index']))
            pp.pprint(each)
    
        return plotDict

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


    def analyzeBloomberg(self,md):


        # Vectorize initial data and show the results
        X,vect = self.vectorizeData(md.masterDF)
        
        # Run LSA on the words and figure out a sub-set of patterns
        svd = TruncatedSVD(n_components=450,random_state=42)
        svd.fit(X)
        print(svd.explained_variance_ratio_)

        self.showComponents(vect.get_feature_names(),svd,30)

        print("=== Variance Explained by top {} components =====".format(len(svd.explained_variance_ratio_)))
        print( sum(svd.explained_variance_ratio_))

        return


    # Takes in the data set, removes outliers, groups headers by day and runs 
    # Vecorize on the information then runs tests of LSA at 
    # different num_components, displays a result of the explained variance
    # so that an optimal num_components can be chosen.
    # returns 
    def analyzeLSA(self,md):


        # Vectorize data without outliers in it
        X,vect = self.vectorizeData(md.masterDF)
        
        # Run LSA on the words and figure out a sub-set of patterns
        svd = TruncatedSVD(n_components=1000,random_state=42)
        X_svd = svd.fit_transform(X)
        s = sum(svd.explained_variance_ratio_)
  
        # Graph the cumulative sum of the explained variance ratio to pick num components
        if(self.showPlots):
            plt.plot(np.cumsum(svd.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance');
            plt.show()

        # For each index create a 3x3 plot that shows the Histogram of the top 9 PC's 
        # Plot the histogram of the values for each document against their label (0 or 1)
        n_comp = 9
        rows,cols=3,3

        for i in md.indexNames:
            labels = md.getLabelsForIndex(i)

            d = labels['y2']
            #d = labels['y1']

            fig = plt.figure(1,figsize=(10,10))
#            fig.subtitle(i)
            plt.grid(True)
            cnt=1
            for i in range(rows):
                for j in range(cols):
                    y = pd.DataFrame(X_svd[:,i+j]) # Create a dataframe with first 10 components
                    y['class'] = d.values

                    pos = y[y['class']==1]
                    neg = y[y['class']==0]

                    plt.subplot(rows,cols,cnt)
                    plt.title("PCA {}".format(cnt))
                    plt.hist(pos.as_matrix()[:,0],stacked=True,color="b",bins=50)
                    plt.hist(neg.as_matrix()[:,0],stacked=True,color="r",alpha=.75,bins=50)
                    cnt+=1
            plt.show()

        print("=== Variance Explained by top {} components =====".format(len(svd.explained_variance_ratio_)))
        print( sum(svd.explained_variance_ratio_))

        # Plot a scatterMatrix of the first 10 components from the LSA analysis with next day close indicated as blue (up or even) red (down)

        from pandas.tools.plotting import scatter_matrix
        pltData = pd.DataFrame(X_svd[:,0:10])        
        colors = ['r','b']
        scatter_matrix(pltData, figsize=(15,15), diagonal='kde',marker='x',c=labels['y2'].apply(lambda x:colors[x]))
        plt.show()

        return

    # Take the data set, remove the outliers, run Sentiment analysis 
    # on each of the headers.  Normalize the sentiments per day
    # Create a final data-feature set that is pos/neg/neu/class
    # For each day.  Graph those values and run a histogram on them showing 
    # Relationship between each histogram and the class variable across 
    # the dataset
    def analysisSentiment(self,md):

        data = md.masterDF.copy()

        # Get the stock data
        pltDict = self.analyzeIndexes(md)
        pltDF = pd.DataFrame(pltDict)

        # Resample the data with last.ffill
        pltDF = pltDF.resample("D").last().ffill()

        # Prebuild the colums with zeros
        for i in indexNames:
            data[i] = 0.0

        # Put together the information by date 
        for d,row in data.iterrows():
            for i in indexNames:
                data.loc[d,i] = pltDF.loc[d,i]

        data['posRoll']=data['pos'].rolling(22).mean()
        data['negRoll']=data['neg'].rolling(22).mean()

        if self.showPlots:
            cnt=0
            fig, axes = plt.subplots(nrows=3, ncols=1,figsize=(12,10))
            for i in indexNames:
                #data[[i,'posRoll','negRoll']].plot(ax=axes[cnt,],secondary_y=[['posRoll','negRoll']])
                data[i].plot(ax=axes[cnt,])
                data['posRoll'].plot(ax=axes[cnt,],secondary_y=True)
                data['negRoll'].plot(ax=axes[cnt,],secondary_y=True)
                axes[cnt,].set_title("Index {}".format(i))

                cnt+=1
            plt.show()

        # Loop through the indexes
        # Create a matrix that has the pos/neg and next day close for each day for that index
        rows,cols=3,2

        plt.figure(1,figsize=(10,10))
        plt.grid(True)
        cnt=1

        for i in indexNames:
            # Pull in the next days closes for the given dates  for that index
            #labels = md.getLabelsForIndex(i)

            #d = labels['y2']
            
            # Put together the information, we do that here so that the document indexes match
            # That way once we convert to the vectorized and transformed data we can re-match
            # back up by row number on the Y class values. 
            #data['class']=-1
            #for d,row in data.iterrows():
            #    data.loc[d,'class'] = d.loc[d,'class']

            posSent = data[['pos',i+'_y2']]
            negSent = data[['neg',i+'_y2']]
            upPos = posSent[posSent[i+'_y2']==1]
            downPos = posSent[posSent[i+'_y2']==0]
            upNeg = negSent[negSent[i+'_y2']==1]
            downNeg = negSent[negSent[i+'_y2']==0]

            # Plot positive Histogram for this index
            plt.subplot(rows,cols,cnt)
            plt.title("{} Positive Sentiment".format(i))
            plt.hist(upPos.as_matrix()[:,0],stacked=True,color="b")
            plt.hist(downPos.as_matrix()[:,0],stacked=True,color="r",alpha=.75)
            cnt+=1
            # Plot negative Histogram for this index
            plt.subplot(rows,cols,cnt)
            plt.title("{} Negative Sentiment".format(i))
            plt.hist(upNeg.as_matrix()[:,0],stacked=True,color="b")
            plt.hist(downNeg.as_matrix()[:,0],stacked=True,color="r",alpha=.75)
            cnt+=1


        plt.show()


if __name__ == "__main__":

    evaluate = Evaluate_Data()

    md = MasterData()
    md.loadData()

    # Pass in the master data set and analys the Index Information
    evaluate.analyzeIndexes(md)

    evaluate.analyzeBloomberg(md)

    evaluate.analyzeLSA(md)
    evaluate.analysisSentiment(md)



     
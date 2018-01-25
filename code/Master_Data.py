import datetime
import glob
import pandas as pd
import pandas_datareader.data as web
import numpy as np
#from textblob import TextBlob
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class MasterData(object):
    
    tmpFiles ='../tmp/'

    srcDir = '../data/scrape_data/*.csv'
    rawDataFile = '../data/data_evaluation_set.csv'

    dataSetFile="../data/index_sentiment_labels_complete.csv"
    


    ## Contstuctor
    def __init__(self):
        self.masterDF = pd.DataFrame()
        self.rawData=pd.DataFrame()
        self.startDate= datetime.datetime(2012,12,20)
        self.endDate = datetime.datetime(2017,12,1)
        self.indexRawData={}
        self.indexNames = ['^GSPC', '^DJI', 'MSCI']
        pass

    # prepData - reads in the source directory files and consolidates the data 
    # Into a single .csv file with the data/data pulled together.
    # This will prevent the evaluation from having to pull all the data together every time it is run

    # Input : srcDir - directory containing all of the source file. FIles in format of index/cdx/data
    # Output : destFile - writes out the list as a .csv dataframe to sourcefile. (rawDataFile)
    def prepRawData(self,srcDir,destFile):

        # Step 1 Read in files from srcDir
        sourceFiles = glob.glob(srcDir)

        #sourceFiles = sourceFiles[:5

        masterDF = pd.DataFrame()
        list = []
        for f in sourceFiles:
            df = pd.read_csv(f,index_col=0)
            s = str(df.iloc[0]['cdx'])[:8]
            d = pd.datetime.strptime(s,'%Y%m%d')
            rows = [[d, data] for data in df['data']]

            list.extend(rows)

        # Createa a dataframe from the list
        df = pd.DataFrame(list,columns=['date','data'])
        df.to_csv(destFile)

        self.rawdata = df

        return self.rawdata


    # Remove all items in the group list from the data object that have a count greater than count
    # input [idx=date][hdr] 1-(n) rows
    # output [idx=date][hdr] 1-(n-outlier) rows 
    def removeOutliers(self,data,count=100):

        data['counter']=1
        group = pd.pivot_table(data, index= ['data'], values= "counter", aggfunc= np.sum,fill_value=0)
        group = group.sort_values('counter',ascending=False)

        # Remember the group dataframe is indexed with the headings, so i== heading, row[0] = count
        for i,row in group.iterrows():
            if(row[0]>count):
                data = data[data['data']!=i].copy()
        return data

        
    # Add in the sentiment information for every row 
    # Input 1 to (n) [date][hdr] --> Output 1 to (n) [date][data][neg][neu][pos][compound]
    # where sentiment is a number between 0 and 1
    def createSentimentData(self,df):

        sid = SentimentIntensityAnalyzer()

        df['neg'] = df.apply(lambda x: sid.polarity_scores(x['data'])['neg'], axis=1)
        df['neu'] = df.apply(lambda x: sid.polarity_scores(x['data'])['neu'], axis=1)
        df['pos'] = df.apply(lambda x: sid.polarity_scores(x['data'])['pos'], axis=1)
        df['compound'] = df.apply(lambda x: sid.polarity_scores(x['data'])['compound'], axis=1)            

        return df

    # Take in the data that is multiple lines per day and consolidate it
    # Input 1 to (n) [date][data][neg][neu][pos][compound]
    # Output 1 to (days) [date][data][neg][neu][pos][compound]
    # Where the sentiment date for each column is the mean of the sentiment for any given day
    # Where the data line is the contatenation of ALL hdrs for the day into one string
    def consolidateData(self,df):
    
        # Pull out the unique days in the set
        indexes = df.index.unique()

        # for each day pull out all the rows and consolidate into a single row
        list =[]
        for i in indexes:
            dfr = df.loc[i]
            hdrs=[]
            for n,j in dfr.iterrows():
                hdrs.append(j.data)

            hdr = " ".join(hdrs)

            negMean = np.mean(dfr['neg'])
            neuMean = np.mean(dfr['neu'])
            posMean = np.mean(dfr['pos'])
            compMean = np.mean(dfr['compound'])
            
            row=[i,hdr,negMean,neuMean,posMean,compMean]
            list.append(row)

        # Add back into the list
        df = pd.DataFrame(list,columns=['date','data','neg','neu','pos','compound'])
        df.set_index('date',inplace=True,drop=True)

        return df

    # pull information for each index in the list of indexNames 
    # input List indexNames
    # Output dict{indexName:dataframe, indexName2:dataframe2}
    # Where dataframe is a data set with the values 
    # Date Open, Low, High, Close, Volume, Adj Close 
    #
    def getIndexData(self,indexNames):

        indexData={}
        for indexName in indexNames:
            df = web.DataReader(indexName, "yahoo", self.startDate, self.endDate)
            indexData.update({indexName:df.copy()})

        return indexData

    # Take a dict of indexes and their data and flatten it into single rows
    # each index becomes a set of columns [date] [<indxname>_Open][<indxname>_Low] .. etc for open,low,high,close,volume,adj close
    def flattenIndexData(self,d,indexNames):

        # Rename the columns for the dataframes based on index name
        list = []
        for i in indexNames:
            d[i].rename(columns={"Open":"{}_Open".format(i),
                                   "Low":"{}_Low".format(i),
                                   "High":"{}_High".format(i),
                                   "Close":"{}_Close".format(i),
                                   "Volume":"{}_Volume".format(i),
                                   "Adj Close":"{}_Adj Close".format(i)},
                                   inplace=True)    
            list.append(d[i])

        # merge all of the dataframes together based on index info
        merged = pd.concat(list,axis=1)

        return merged

    # merge together the index trading information and consolidated data
    # input
    # index [date][indx_Open][indx_Low][indx_High][indx_Close][indx_Volume][indx_Adj Close]...[all indexes]
    # consData [date][data][data][neg][neu][pos][compound]
    # Output index+consData
    def mergeIndexAndConsolidated(self,indexDF,consDF):
        merged = pd.concat([consDF,indexDF],axis=1,join="inner")
        return merged

    # addLabels takes the merged data set and adds apprpriate labels for each index
    # each index will get 2 new columns, <index>_y1, <index>_y2  
    # where y1 is the current days market close being up or down (0 or 1)
    # and y2 is the next trading day market close being up or down (0 or 1)
    # those labels can be used by a final ML algorithm for supervised learning
    def addLabels(self,df,indexNames):

        # for each index pull out the column for it's adj close
        # for each day add a column for today's day vs. prev day close as 0 or 1
        # and next day vs today adj close as 0 or 1
        df = df.reset_index() # Reset the index so that it's not using the dates
        df = df.rename(columns={"index":"Date"})
        for i in indexNames:
            df[i+"_y1"]=0
            df[i+"_y2"]=0
            #adjCloseDF.reset_index()    # reindex so we can index in by number+1,-1 etc.
            for n in range(1,len(df)-1):
                if df.loc[n,i+"_Adj Close"] >= df.loc[n-1,i+"_Adj Close"]:
                    df.loc[n,i+"_y1"]=1
                else:
                    df.loc[n,i+"_y1"]=0

                if df.loc[n+1,i+"_Adj Close"] >= df.loc[n,i+"_Adj Close"]:
                    df.loc[n,i+"_y2"]=1
                else:
                    df.loc[n,i+"_y2"]=0    

        df = df.set_index('Date')

        return df


    # Create Data creates the large data-set from scratch and pulls it in from 
    # all of the primary sources
    def createData(self):

        # First try and load the raw data file
        try:
            self.rawData = pd.read_csv(self.rawDataFile,index_col=0,parse_dates=[1],dtype={"data":"str"})  # Parse the second column as dates
            self.rawData.set_index('date',inplace=True,drop=True)
        except :
            print("Could not find Raw Data File")
        
        if len(self.rawData)<1:
            self.prepRawData(self.srcDir,self.rawDataFile)

        ### Raw Data is now 1--(n) rows [Date][Data (hdr)] where Date is the date of the header and data is a single header 
        # Make a local copy here for use without affecting Class scoped
        rawData=self.rawData[:].copy()  #when testing limit this to 5000

        # Remove all of the outliers, any entry that shows up over 200 times
        noOutliersDF = self.removeOutliers(rawData,200)
        
        # Add on the sentiment information for every header
        ### sentimentDF is 1--(n) rows [Date][Data(hdr)][neg][neu][pos][compound]
        sentimentDF = self.createSentimentData(noOutliersDF)
        sentimentDF.to_csv(self.tmpFiles+"tmp1_sentiment.csv")

        # Consolidate the data into 1 row per day, taking the average neg/neu/pow/consolidated per day
        # Also consolidate the data/header information so that it is ALL the headers in the string for that
        # given day. 
        consolidatedDF = self.consolidateData(sentimentDF)        
        consolidatedDF.to_csv(self.tmpFiles+"tmp2_consolidated.csv")
        
        # Pull the index information for the indexes.  This data is ffilled for the full
        # date range
        self.indexRawData = self.getIndexData(self.indexNames)

        # Flatten IndexData
        self.indexData = self.flattenIndexData(self.indexRawData,self.indexNames)
        self.indexData.to_csv(self.tmpFiles+"tmp3_indexdata.csv")

        # Resample the IndexData so that it covers all days
        self.indexData = self.indexData.resample("D").last().ffill()

        # Add the stock information to the data.  This will add the open-low-high-adj close for each index for each day
        # format is <index>_open, <index>_low, etc...
        mergedDF = self.mergeIndexAndConsolidated(consolidatedDF,self.indexData)
        mergedDF.to_csv(self.tmpFiles+"tmp4_mergeddata.csv")

        # Finally add in the current day and next day classification binaries.
        # 0 if the current or next ay close is lower than the previous days
        # 1 if the current or next day close is higher or equal than the previous days
        # note that there are entries for EACH index <index>_currenday_class,<index>_nextday_class
        finalDF = self.addLabels(mergedDF,self.indexNames)

        # Remove the first and last entries as they don't have labels
        masterDF = finalDF[0:-1]

        masterDF.to_csv(self.dataSetFile)


    # Return the NP's for each index and the y1 and y2 values
    # the NP is simply the % of TP/total for y1 and y2 it represents what would happen 
    # if you always chose UP for the day or next day's market
    # Return a dataframe with index,y1_perc,y2_perc for each row
    def getNaivePredictors(self):

        list=[]
        for i in self.indexNames:
            tp_y1 = float(sum(self.masterDF[i+"_y1"]))
            tp_y2 = float(sum(self.masterDF[i+"_y2"]))
            total = float(len(self.masterDF))
            list.append([i,tp_y1/total,tp_y2/total])

        return pd.DataFrame(list,columns=['index','y1_perc','y2_perc'])


    # Read in the data from the file, set it up with proper colnames and use Date as index
    # Other classes only need to call loadData to get it and then use the 
    # data accessor methods to pull out slices. 
    def loadData(self):
        self.masterDF=pd.read_csv(self.dataSetFile,index_col=0,parse_dates=[0])
        return self.masterDF

    # getIndexInfo returns just the information for a specific index as a dataframe
    # input: indexName
    # Output: DataFrame with [Date][Open][Low][High][Close][Volume][Adj Close]
    #
    def getIndexInfo(self,indexName):

        # Create the column names
        cols = [indexName+"_Open",indexName+"_High",indexName+"_Low",indexName+"_Close",indexName+"_Volume",indexName+"_Adj Close"]
        df = self.masterDF[cols]
        return df

    # Return a DF with the y1 and y2 labels for a given index
    # Note the labels are renamed so that the index name is stripped out
    def getLabelsForIndex(self,indexName):
        cols = [indexName+"_y1",indexName+"_y2"]
        df = self.masterDF[cols]
        df = df.rename(columns={indexName+"_y1": 'y1', indexName+"_y2": 'y2'})
        return df

    # Return a DF for the semantic information for a specific index.  Index name is stripped out
    # Returns [neg][neu][pos][compount] for all rows
    def getSemanticInfo(self):
        cols = ["neg","neu","pos","compound"]
        df = self.masterDF[cols]
        return df

if __name__ == "__main__":

    md = MasterData()
    md.createData()
    md.loadData()
    iDF = md.getIndexInfo(md.indexNames[0])
    pass







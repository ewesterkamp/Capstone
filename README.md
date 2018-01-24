# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Proposal and Capstone Project

# Versions
Code primarily run with Python 2.7.13 however `Evaluate_Data.py` also run with Python 3.5.4 when *MemoryError* encountered. 

# Running Scripts
`cd .\code`

`python Master_Data.py` - Reads in raw files from ../data/scrape_data and creates master data-set

`python Evaluate_Data.py` - Creates all charts and statstic information

`python Benchmark_Mode.py` - Creates the benchmark metrics for each of the indexes

`python LSA_SVC_RF_TimeSeries_Model.py` - Runs Model for LSA with RandomForest all window sizes

`python LSA_SVC_RF_TimeSeries_Model.py SVC` - Runs model for LSA with SVC all window sizes

`python Semantic_SVC_RF_TimeSeries_Model.py` - Runs model for Semantic with RandomForest all window sizes

`python Semantic_SVC_RF_TimeSeries_Model.py SVC` - Runs model for Semantic with SVC all window sizes
 


## Directories and Files ##

# ./

**Machine Learning Nanodegree - Predicting Market Indexes.pdf** - Machine Learning Capstsone Project final report. 

**mectrics\_summary.xlsx** - Contains the aggregated metrics information from the ./data and ./tmp files for the final report.  Tab **metrics\_summary** contains primary information.  **metrics\_summary\_bak** has the True Positive, True Negative, False Positive and False Negative data included.    

**Market Index Prediction with Sentiment Analysis - Proposal.pdf** - Original project proposal. 


# ./code
Contains the python scripts and runable scripts for project. 

**bloomberg\_spider.py** - scraper file that extracts the header files from Archive.org.  Puts files into the data/scrape_data folder.  Must be run with the Scrapy python toolkit installed separately.  No need to run this file as a test as all source files created by it are included in this project in the ./data/scrape\_data/ folder.

**Master\_Data.py** - Python file that can either be run stand-alone or loaded as part of another python script.  It looks to see if the master data set file exists, if it does it loads it into a Dataframe and makes it available to other objects.  If it doesn't it utilizes the output form bloomberg_spider.py to create the data and makes calls out to finance.yahoo.com to pull in the index information.  

**Evaluate\_Data.py** - This is the source code for creating many of the initial evaluations, charts and views of the initial data set.  Initially looked at raw data but has been rebuilt to use the Master_Data.py file for simplicity. 

**LSA\_SVC\_RF\_TimeSeries\_Model.py** - Python file that runs the LSA analysis on the information from the MasterData class.  It then runs both SVC and RandomForest classifiers on the data for the specified time-series window lengths.  Passing in the argument SVC runs the script for the SVC classifier, default behavior is RandomForest.  

`Ex.  python LSA_SVC_RF_TimeSeries_Model.py SVC - runs as SVC`

`Ex.  python LSA_SVC_RF_TimeSeries_Model.py - runs as RandomForest`

**Semantic\_SVC\_RF\_TimeSeries\_Model.py** - Similar to the LSA file but feeds in the semantic data directly from the MasterData class but also runs SVC and RandomForest classifiers on the data for the specified windows.  Same SVC paramaters as LSA_SVC_RF_TimeSeriese_Model functionality. 

`Ex.  python Semantic_SVC_RF_TimeSeries_Model.py SVC - runs as SVC`

`Ex.  python Semantic_SVC_RF_TimeSeries_Model.py - runs as RandomForest`
 
**Benchmark\_Model.py** - Python script that creates a the benchmark metrics for the 3 indexes. 


# ./data
The data folder contains the primary output files for the project.  

**index\_sentiment\_labels\_complete.csv** - Data file that is created by and also loaded by the MasterData class file.  Contains all the daily information including fully headers for each day, index information for each index, sentiment data and the labels for each index per day. 

**./data/scrape\_data/\*.csv** - The individual scraped data files created by bloomberg__spider.py

**data\_evaluation\_set.csv** - intermediate file created when running MasterData.createData that contains the consolidated information from the scrape_data files.  

**LSA\_RandForest\_metrics.csv** - Metrics file for LSA RandomForest run with all windows and indexes

**^DJI\_20\_lsa\_RF\_predictions\_conclusions.xlsx** - Summary version of file from ./tmp folder.  Comparison of the benchmark vs. Buy and Hold model data is in this file as well as final chart.  Portfolio simulation created for each unique trading day where 50% of shares sold off if next day is down.  And all cash used to buy shares if next day is up.   
 
**benchmark\_naive\_predictor.csv** - Benchmark data for the three indexes. 


# ./tmp
The tmp folder contains temporary output files that are create by the python scripts.  For instance MasterData creates a number of temporary output files during the data creation phase so that the data could be reviewed and validated at each point.

**tmp1\_sentiment.csv** - Date, Bloomberg Headers and Sentiment information

**tmp2\_consolidated.csv** - Date, Bloomberg Headers consolidated for the day and sentiment information

**tmp3\_indexdata.csv** - Date, Daily index data

**tmp4\_mergerdata.csv** - Merge of tmp3_indexdata with tmp2_consolidated 

**<Index_name\>\_#\_lsa\_RF\_Predictions** - temporary files that contain the output of the predictions for LSA and RandomForest for each window and index.  ^DJI\_20\_lsa\_RF\_Predictions is the source file used to create the Benchmark vs. Model in Section V. Conclusions. 

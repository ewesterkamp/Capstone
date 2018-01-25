import scrapy
import datetime
import pandas as pd
import numpy as np


# spider crawls all files coming in from the cdx search on bloomberg at archive.org
# creates a .csv file in the folder scrape_data, 1 file for each page
# file is named by the csv data name {cdxtimestamp}.csv
# there can be multiple files per day and some days without any files for given years
#
class bloomberg_spider(scrapy.Spider):
    name = "bloomberg"
    srcFile = ('bloomberg_cdx_dedupe.csv')

    def start_requests(self):

        baseUrl = 'http://web.archive.org/web/{cdxdate}/http://www.bloomberg.com/'

        # Read in the file containing all the CDX index codes 
        df = pd.read_csv(self.srcFile)
        print df.head()

        cd = df['cdxtimestamp']
        cdxDates = cd  # slice out the last 3
        print type(cdxDates), len(cdxDates)

        urls=[]
        for d in cdxDates:
            urls.append(baseUrl.format(cdxdate=d))

        # request each url for parsing
        # Open up the url file
        #urls = urls[:500]       ## Reduce to 100 for testing
        for url in urls:
            print "====> Processing URL {}  ===== ".format(url)
            yield scrapy.Request(url=url, callback=self.parse)


    # Extract the CDX timestamp form the URL
    def urlToCDXDate(self,url):
        tokens = url.split("/")
        cxDate = tokens[4]
        return cxDate

    def parse(self, response):

        cdx = self.urlToCDXDate(response.url)
        cdx = cdx.encode('ascii', 'ignore')
        print "==== Bloomberg Scraper Parse called on {} ====".format(cdx)

        items = []
        for a in response.css('a'):
            try:
                txt = a.css("::text").extract_first()
                txt = txt.encode('ascii','ignore')
                txt = txt.lstrip()
                txt = txt.rstrip()
                if(len(txt.split())>3 and (len(txt)>0)):
                    items.append(txt)
            except:
                pass
        
        #df = pd.DataFrame([[0,"test"]],columns=['cdxdatetime','header'])

        data = np.array([[cdx,items[0]]])
        #data = np.empty((len(items),2))
        print data.shape
        print data
        for i,v in enumerate(items):
            # skip first item
            if i==0:
                pass
            else:
                row = [cdx,v]
                data = np.append(data,[[cdx,v]],axis=0)

        df = pd.DataFrame(data, columns=['cdx','data'])
        print df.head()
        df.to_csv('./scrape_data/'+cdx+'.csv')

        print "==== Bloomberg Scraper Parse finished on {} ====".format(cdx)


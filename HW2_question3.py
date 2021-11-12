#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import groupby


# ## Question 3

# #### part a

# In[4]:


# read files
df2011 = pd.read_sas("C:/Users/fengy/Desktop/stats 507/hw2/DEMO2011-2012.XPT")
# add columns to a single dataframe
df2011['year'] = "2011-2012"
# read file and import it as df
df2013 = pd.read_sas("C:/Users/fengy/Desktop/stats 507/hw2/DEMO2013-2014.XPT")
# add columns to a single dataframe
df2013['year'] = "2013-2014"
# read file and import it as df
df2015 = pd.read_sas("C:/Users/fengy/Desktop/stats 507/hw2/DEMO2015-2016.XPT")
# add columns to a single dataframe
df2015['year'] = "2015-2016"
# read file and import it as df
df2017 = pd.read_sas("C:/Users/fengy/Desktop/stats 507/hw2/DEMO2017-2018.XPT")
# add columns to a single dataframe
df2017['year'] = "2017-2018"
# merge seperate dataframes into one
mergedf = pd.concat([df2011, df2013,df2015,df2017])
# select columns from a dataframe
mydf = pd.DataFrame(mergedf[["SEQN","RIDAGEYR","RIDRETH3","DMDEDUC2","DMDMARTL",
                  "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR","year"]])
# change the column name
mydf = mydf.rename({'SEQN': 'id', 'RIDAGEYR': 'age',
                'RIDRETH3':'race', 'DMDEDUC2':'education',
                 'DMDMARTL':'marital', 'RIDSTATR':'interview way',
                 "SDMVPSU": "variance pseudo",
                 "SDMVSTRA":"variance estimation",
                 "WTMEC2YR":"mec interview weight",
                 "WTINT2YR":"intverview weight",
                 "year":"year"
                }, axis='columns')
# change the column data types
mydf = mydf.astype({"id": int, "age": int,"race":int, 
                    "education":"category",
                    "marital":'category'})
display(mydf.head(10))
# display the dataframe
display(mydf)


# #### part b

# In[5]:


# +
# read dataframe,select required columns, add columns, and build it as a single df
oral2011 = pd.read_sas("C:/Users/fengy/Desktop/stats 507/hw2(1)/OHXDEN2011-2012.XPT")
oral2011 =oral2011.loc[:,'SEQN':'OHX31CTC']
oral2011.drop(['OHDEXSTS', 'OHXIMP'],inplace=True, axis=1)
oral2011["year"] = "2011-2012"


# read dataframe,select required columns, add columns, and build it as a single df
oral2013 = pd.read_sas("C:/Users/fengy/Desktop/stats 507/hw2(1)/OHXDEN2013-2014.XPT")
oral2013 =oral2013.loc[:, 'SEQN':'OHX31CTC']
oral2013.drop(['OHDEXSTS', 'OHXIMP'], inplace=True, axis=1)
oral2013["year"] = "2013-2014"


# read dataframe,select required columns, add columns, and build it as a single df
oral2015 = pd.read_sas("C:/Users/fengy/Desktop/stats 507/hw2(1)/OHXDEN2015-2016.XPT")
oral2015 =oral2015.loc[:, 'SEQN':'OHX31CTC']
oral2015.drop(['OHDEXSTS', 'OHXIMP'], inplace=True, axis=1)
oral2015["year"] = "2015-2016"


# read dataframe,select required columns, add columns, and build it as a single df
oral2017 = pd.read_sas("C:/Users/fengy/Desktop/stats 507/hw2(1)/OHXDEN2011-2012.XPT")
oral2017 =oral2017.loc[:,'SEQN':'OHX31CTC']
oral2017.drop(['OHDEXSTS', 'OHXIMP'], inplace=True, axis=1)
oral2017["year"] = "2017-2018"

# merge each sepearte df into one, change the datatype, and change the column name.
partb = pd.concat([oral2011, oral2013,oral2015,oral2017])
partb = partb.astype({"SEQN": int})
partb = partb.rename({"SEQN": 'id', "OHDDESTS": 'status'},
                   axis='columns')
partb = partb.rename(columns=lambda x: x[3:])
partb = partb.rename({'':'id', 'tus':'status','r':'year'},
                     axis='columns')
display(partb)
# -


# #### part c

# In[6]:


a = mydf['id']
b = partb['id']
# find out the cases shareing the same Id
display(len(set(a).intersection(set(b))))


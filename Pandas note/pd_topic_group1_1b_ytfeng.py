
# Wenxiu Liao
# wenxiul@umich.edu

# ### sort_values()
# * The Series.sort_values() method is used to sort a Series by its values.
# * The DataFrame.sort_values() method is used to sort a DataFrame by its column or row values.
#     - The optional by parameter to DataFrame.sort_values() may used to specify one or more columns to use to determine the sorted order.

import numpy as np
import pandas as pd
from os.path import exists
from scipy import stats
from scipy import stats as st
from warnings import warn
from scipy.stats import norm, binom, beta
import matplotlib.pyplot as plt

df1 = pd.DataFrame(
    {"A": [2, 1, 1, 1], "B": [1, 3, 2, 4], "C": [5, 4, 3, 2]}
)
df1.sort_values(by="C")

# The by parameter can take a list of column names, e.g.:

df1.sort_values(by=["A", "B"])

# You can specify the treatment of NA values using the na_position argument:

s = pd.Series(
    ["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"],
    dtype="string"
)
s.sort_values(na_position="first")


#!/usr/bin/env python
# coding: utf-8

# ## Topics in Pandas<br>
# **Stats 507, Fall 2021**

# ## Contents<br>
# + [pandas.cut function](#pandas.cut-function)
# + [Sampling in Dataframes](#Sampling-in-Dataframes)
# + [Idioms-if/then](#Idioms-if/then)

# ___<br>
# ## pandas.cut function<br>
# **Name: Krishna Rao**<br>
# <br>
# UM-mail: krishrao@umich.edu

# ## pandas.cut function<br>
# * Use the cut function to segment and sort data values into bins. <br>
# * Useful for going from a continuous variable to a categorical variable. <br>
# * Supports binning into an equal number of bins, or a pre-specified array of bins.<br>
# <br>
# #### NaNs?<br>
# * Any NA values will be NA in the result. <br>
# * Out of bounds values will be NA in the resulting Series or Categorical object.

# ## Examples<br>
# * Notice how the binning start from 0.994 (to accommodate the minimum value) as an open set and closes sharply at 10<br>
# * The default parameter 'right=True' can be changed to not include the rightmost element in the set<br>
# * 'right=False' changes the bins to open on right and closed on left

# In[2]:


import pandas as pd
import numpy as np
input_array = np.array([1, 4, 9, 6, 10, 8])
pd.cut(input_array, bins=3)
#pd.cut(input_array, bins=3, right=False)


# +<br>
# Observe how 0 is converted to a NaN as it lies on the open set of the bins<br>
# 1.5 is also converted to NaN as it lies between the sets (0, 1) and (2, 3)

# In[3]:


bins = pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
#bins = [0, 1, 2, 3, 4, 5]
pd.cut([0, 0.5, 1.5, 2.5, 4.5], bins)
# -


# ## Operations on dataframes<br>
# * pd.cut is a very useful function of creating categorical variables from continous variables<br>
# * 'bins' can be passed as an IntervalIndex for bins results in those categories exactly, or as a list with continous binning.<br>
# * Values not covered by the IntervalIndex or list are set to NaN.<br>
# * 'labels' can be specified to convert the bins to categorical type variables. Default is `None`, returns the bins.

# ## Example 2 - Use in DataFrames<br>
# * While using IntervalIndex on dataframes, 'labels' can be updated with pd.cat.rename_categories() function<br>
# * 'labels' can be assigned as string, numerics or any other caregorical supported types

# +

# In[4]:


df = pd.DataFrame({"series_a": [0, 2, 1, 3, 6, 4, 2, 8, 10],
                   "series_b": [-1, 0.5, 2, 3, 6, 8, 14, 19, 22]})


# In[5]:


bin_a = pd.IntervalIndex.from_tuples([(0, 2), (4, 6), (6, 9)])
label_a = ['0 to 2', '4 to 6', '6 to 9']
df['bins_a'] = pd.cut(df['series_a'], bin_a)
df['label_a'] = df['bins_a'].cat.rename_categories(label_a)


# In[6]:


bin_b = [0, 1, 2, 4, 8, 12, 15, 19]
label_b = [0, 1, 2, 4, 8, 12, 15]
df['bins_b'] = pd.cut(df['series_b'], bin_b)
df['labels_b'] = pd.cut(df['series_b'], bin_b, labels=label_b)


# In[7]:


df
# -


# #### References:<br>
# * https://pandas.pydata.org/docs/reference/api/pandas.cut.html<br>
# * https://stackoverflow.com/questions/55204418/how-to-rename-categories-after-using-pandas-cut-with-intervalindex<br>
# ___

# ___<br>
# ## Sampling in Dataframes<br>
# **Name: Brendan Matthys** <br>
# <br>
# UM-mail: bmatthys@umich.edu

# ## Intro -- df.sample<br>
#

# Given that this class is for an applied statistics major, this is a really applicable topic to be writing about. This takes a dataframe and returns a random sample from that dataframe. Let's start here by just importing a dataframe that we can use for

# In[8]:


import pandas as pd
import os
import pickle
import numpy as np


# # +<br>
# -----------------------------------------------------------------------------

# In[9]:


filepath =os.path.abspath('')
if not os.path.exists(filepath + "/maygames"):
    nba_url ='https://www.basketball-reference.com/leagues/NBA_2021_games-may.html'
    maygames = pd.read_html(nba_url)[0]
    maygames = maygames.drop(['Unnamed: 6','Unnamed: 7','Notes'], axis = 1)
    maygames = maygames.rename(columns =
                               {
        'PTS':'Away Points',
        'PTS.1':'Home Points'
    })

    #dump the data to reference for later
    pickle.dump(maygames,open(os.path.join(filepath,'maygames'),'wb'))
else:
    maygames = pd.read_pickle('maygames')
    
maygames
# -


# The dataframe we will be working with is all NBA games from the 2020-2021 season played in May. We have 173 games to work with -- a relatively strong sample size.

# Let's start here with taking a sample with the default parameters just to see what the raw function itself actually does:

# In[10]:


maygames.sample()


# The default is for this function to return a single value from the dataframe as the sample. Using the right parameters can give you exactly the sample you're looking for, but all parameters of this function are optional.

# ## How many samples?

# The first step to taking a sample from a population of data is to figure out exactly how much data you want to sample. This function has two different ways to specify this -- you can either use the parameters n or frac, but not both.<br>
# <br>
# ### n <br>
#  * This is a parameter that takes in an integer. It represents the numebr of items from the specified axis to return. If neither n or frac aren't specified, we are defaulted with n = 1.<br>
#  <br>
# ### frac<br>
#  * This is a parameter that takes in a float value. That float returns the fraction of data that the sample should be, representative of the whole population. Generally speaking, the frac parameter is usually between 0 and 1, but can be higher if you want a sample larger than the population<br>
#  <br>
# ### Clarification <br>
# It's important to note that if just any number is typed in, the sample function will think that it is taking an input for n.

# In[11]:


maygames.sample(n = 5)


# In[12]:


maygames.sample(frac = 0.5)


# In[13]:


print(len(maygames))
print(len(maygames.sample(frac = 0.5)))


# ## Weights and random_state

# The weights and random_state paramteres really define the way that we are going to sample from our original dataframe. Now that we have the parameter that tells us how many datapoints we want for our sample, it is imperative that we sample the right way. <br>
# <br>
# ### Weights<br>
# <br>
# Weights helps define the probabilities of each item being picked. If the parameter is left untouched, then the default for this is that all datapoints have an equal probability of being chosen. You can choose to specify the weights in a variety of ways. <br>
# <br>
# If a series is used as the parameter, the weights will align itself with the target object via the index.<br>
# <br>
# If a column name is used, the probabilities for being selected will be based on the value of that specific column. If the sum of the values in that column is not equal to 1, the weights of those values will be normalized so that they sum to 1. If values are missing, they will be treated as if they are weighted as 0.

# In[14]:


maygames.sample(n = 10, weights = 'Attend.')


# The sample above took in 10 datapoints, and was weighted based on the game attendance, so that the games with more people at them had a higher chance of being picked.

# ### Random_state<br>
# <br>
# Random state is essentially the parameter for the seed we want. This creates a sample that is reproducible if you want it to be. Generally, an integer is inputted for the parameter, but an np.random.RandomState object can be inserted if wanted. The default value for this is None.

# In[15]:


sample_1 = maygames.sample(n = 10, weights = 'Attend.', random_state = 1)
sample_1


# In[16]:


sample_2 = maygames.sample(n = 10, weights = 'Attend.', random_state = 1)
sample_2


# In[17]:


sample_1 == sample_2


# As you can see, the random_state parameter creates a sample that can be reproduced for future uses, which can prove to be incredibly helpful.

# ## Replace and ignore index

# The last few optional parameters we have are replace and ignore index. Both can be advantageous in their own right. <br>
# <br>
# ### Replace<br>
# <br>
# The parameter replace specifies whether we want to be able to sample with or without replacement. It takes in a Boolean as input. If True, then the datapoint has the ability to be chosen again into the sample. If False, the datapoint is removed from the pool of possible points to be chosen.

# In[18]:


maygames.sample(
    n = 10,
    weights = 'Attend.',
    random_state = 1,
    replace = True)


# ### Ignore_index<br>
# <br>
# The ignore_index parameter is useful if you want your index to be relabeled instead of having the original index labels in the sample. This takes in a Boolean input. If True, the resulting index is relabeled, but if False (default), then the resulting index stays how it was.

# maygames.sample(<br>
#     n = 10,<br>
#     weights = 'Attend.',<br>
#     random_state = 1,<br>
#     replace = True,<br>
#     ignore_index = True)<br>
# --

# ___<br>
# ## Idioms-if/then<br>
# **Name: Junqian Liu**<br>
# <br>
# UM-mail: junqianl@umich.edu

# In[21]:


import numpy as np
import pandas as pd


# ## Dataframe method
# - The dataframes allows us to change the values of one or more columns directly by the conditions
# - df.loc allows us to choose which columns to work as the condition, and which columns to be changed based on the conditions
# - More specifically, it works as df.loc[conditions, target columns] = values

# In[22]:


df = pd.DataFrame(
    {"apple": [4, 5, 6, 7], "boy": [10, 20, 30, 40], "cat": [100, 50, -30, -50],
    "dog": [3, 5, 0, 6]}
)
df.loc[df.apple >= 5, "boy"] = -1
df.loc[df.apple >= 5, ["cat", "dog"]] = 555
df


# ## Pandas method
# - pandas also can achieve the same aim by setting up a mask
# - pandas.DataFrame.where allows to decide if the conditions are satisfied and then change the values
# - overall, the goal is achieved by setting up the mask to the dataframe and using pandas.DataFrame.where to replace the values.
# - needs to assign to the dataframe after replacing values

# In[23]:


df2 = pd.DataFrame(
    {"apple": [4, 5, 6, 7], "boy": [10, 20, 30, 40], "cat": [100, 50, -30, -50],
    "dog": [3, 5, 0, 6]}
)
df_mask = pd.DataFrame(
    {"apple": [True] * 4, "boy": [False] * 4, "cat": [True, False] * 2,
    "dog": [False] * 4}
)
df2 = df2.where(df_mask, 818)
df2


# ## Numpy method
# - Similar to pandas method, np.where can also replace the value through if/then statement
# - It is more convenience as it doesn't need to set up the masks
# - It works by np.where(condistions, if true, else), to be more specific, the example is given below

# In[24]:


df3 = pd.DataFrame(
    {"apple": [4, 5, 6, 7], "boy": [10, 20, 30, 40], "cat": [100, 50, -30, -50],
    "dog": [3, 5, 0, 6]}
)
df3["elephant"] = np.where(df["apple"] > 5, "water", "banana")
df3

# -*- coding: utf-8 -*-
# ## Topics in Pandas
# **Stats 507, Fall 2021**
#

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using
# the exact title with spaces replaced by a dash.
#
# + [Filling missing values](#Filling-missing-values)
# + [Missing values in pandas](#Missing-values-in-pandas)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Filling missing values
# ## Zane Zhang  zzenghao@umich.edu

# + [markdown] slideshow={"slide_type": "fragment"}
# > Creat a dataframe with nan value

# + slideshow={"slide_type": "fragment"}
import pandas as pd
import numpy as np

df = pd.DataFrame(
    np.random.randn(5, 3),
    index=["a", "c", "d", "e", "f"],
    columns=["one", "two", "three"],
)

df=df.reindex(["a", "b", "c", "d", "e", "f"])
df

# + [markdown] slideshow={"slide_type": "slide"}
# ## filna() method
# * fillna() can “fill in” NA values with non-NA data in a couple of ways
#     * Replace NA with a scalar value
#
#
# **fill the nan value with -1**

# + slideshow={"slide_type": "fragment"}
df.fillna(-1)

# + [markdown] slideshow={"slide_type": "subslide"}
# **fill nan with string**

# + slideshow={"slide_type": "fragment"}
df.fillna("missing")

# + [markdown] slideshow={"slide_type": "slide"}
# ## filna() method
# * fillna() can “fill in” NA values with non-NA data in a couple of ways
#     * Fill gaps forward(method="Pad") or backward(method="bfill")

# + slideshow={"slide_type": "fragment"}
print("fill the data based on the forward data")
print(df.fillna(method="pad"))
print("fill the data based on the backward data")
print(df.fillna(method="bfill"))
# -

# ## Missing values in pandas
# ## Mohammad Zhalechian  mzhale@umich.edu

# * Panda is flexible with regard to handling missing data
# * $NaN$ is the default missing value marker in Pandas
# * Pandas provides two function $insa()$ and $notna()$ to detect missing values

# +
df = pd.DataFrame({'one': [1,2,3], 'two':['a','b','c']})
df2 = df.reindex([0,1,2,3,4])

pd.isna(df2)
pd.notna(df2)
pd.isna(df2['two'])
# -

# ## Inserting Missing Values
#
# * We can insert missin values using $None$ or $numpy.nan$.
# * Pandas objects provide compatibility between $None$ and $numpy.nan$.

# +
df2.loc[0,'one'] = np.nan
df2.loc[1,'two'] = None

pd.isna(df2)
# -

# ## Calculations with Missing Values
#
# * Most of descriptive statistics and computational methods are written to account for missing data
# * For example:
#     * When summing data (e.g., $np.sum()$), missing values will be treated as zero.
#     * Cumulative methods like $cumsum()$, $np.mean()$, $cumprod()$ ingnore the missing values be default. We can use $skipna=False$ to override this behavior.

np.sum(df2['one'])
np.mean(df2['one'])

# ## Filling missing values
#
# * We can fill missing values using several methods
#     * Replace missing values with a scalar value using $df.fillna('name')$.
#     * Filling the missing values with non-missing values forward or backward using $df.fillna(method = 'pad')$.

# +
df3= df2.copy()
df3.fillna('missing')

df4= df2.copy()
df4.fillna(method = 'pad')
# -




#!/usr/bin/env python3

# importing packages
from IPython.display import HTML
import pandas as pd 
import numpy as np
import os 
from scipy import stats
from scipy.stats import chi2_contingency
from collections import defaultdict
from scipy.stats import norm, inom, beta
import re 

# Andrew Heldrich
# aheldric@umich.edu

# ## `rank()` Method
# - A common need is to rank rows of data by position in a group
# - SQL has nice partitioning functions to accomplish this, e.g. `ROW_NUMBER()`
# - Pandas `rank()` can be used to achieve similar results

# ## Example
# - If we have sales data for individual people, we may want to find their sales
# rank within each state

rng = np.random.default_rng(10 * 8)
df = pd.DataFrame({"id":[x for x in range(10)],
                    "state":["MI", "WI", "OH", "OH", "MI", "MI", "WI", "WI",
                                "OH", "MI"],
                    "sales":rng.integers(low=10, high=100, size=10)})
df

# ## groupby -> rank
# - Need to chain `groupby()` and `rank()`
# - Assign results to new ranking column
# - Now we can easily see the best sales person in each state and can do
# additional filtering on this column
# - Especially useful for time-based ranking
#     - E.g. Find last 5 appointments for each patient

df["sales_rank"] = (df.groupby("state")["sales"]
                        .rank(method="first", ascending=False))
df

top_2 = (df.query('sales_rank < 3')
            .sort_values(by=["state", "sales_rank"]))
top_2


# # name: Siwei Tang Email: tangsw@umich.edu
# # Q0
# ## Time series/ data functionality
#
# The Python world has a number of available representation of dates, times, deltas, and timespans. Whiles the times series tools provided by Pandas tend to be the most useful for data science applications, it's helpful to see their relationsip to other packages used in Python.
#
# ## Native Python dates and times: `datetime` and `dateutil`
#
# Pythonn's baseic objects for working with dates and times reside in the built-in `dateime` module. Along with the third-party `dateutil` module, you can use it to quickly perform a host of useful functionalities on dates and time. 

# - build a date using the `datetime` type

from datetime import datetime
datetime(year = 2021, month=10, day=20)

# - using dateutil module to parse dates from a variety of strng formats

from dateutil import parser
date = parser.parse("20th of October, 2021")
date 

# - Once you have a `datetime` object, you can do things like printing the day of the week:

date.strftime('%A')

# In the final line, `%A` is part of the [strfyime section](https://docs.python.org/3/library/datetime.html) od Python's [datetime documentation]()

# ## Typed arrays of times: Numpy's `datatime64`
# - The `datatime64` dtype encoded dates as 64-bit inegers, and thus allows arrays of dates to be represented very compactly. The `datatime64` requires a very specific input format:

date =np.array('2021-10-20', dtype=np.datetime64)
date

# - Once we have this date formated, however, we can quickly do vectorized operations on it

date + np.arange(12)

# - One detail of the `datetime64` and `timedelta64` object is that they are build on a fundamental time unit. Because the `datetime64` object is limited to 64-bit precision, the range of encodable times is $2^{64}$ times this fundamental unit. In other words, `datetime64` imposes a trade-off between **time resolution** and **maximum time span**.

# ## Dates and times in pandas: best of both worlds
# Pandas builds upon all the tools just discussed to provide a `Timestamp` object, which combines the ease-of-use of `datetime` and `dateutil` with the efficient storage and vectorized interface of `numpy.datetime64`. From a group of these `Timestamp` objects, Pandas can construct a `DatetimeIndex` that can be used to index data in a `Series` or `DataFrame`.

date = pd.to_datetime('20th of October, 2021')
date

date.strftime('%A')

# - we can do Numpy-style vectorized operations directly on this same object:

date + pd.to_timedelta(np.arange(12),'D')

# ## Pandas Time Series Data Structures
# - for time stamps, Pandas provides the `Timestamp` type. As mentioned before, it is essentially a replacement for Python's native `datetime`, but is based on the more efficient `numpy.datetime64` date type. The associated Index structure is `DatetimeIndex`. 
# - for time Periods, Pandas provides the `Period` type. This encodes a fixed-frequency interval based on `numpy.datetime64`. The associated index structure is `PeriodIndex`.
# - For time deltas or durations, Pandas provides the `Timedelta` type. `Timedelta` is a more efficient replacement for Python's native `datetime.timedelta` type, and is based on `numpy.timedelta64`. The assocaited index structure is `TimedeltaIndex`.
#
# Passing a single date to `pd.to_datetime()` yields a `Timestamp`; passing a series of dates by default yields a `DatetimeIndex`:

dates = pd.to_datetime([datetime(2021,10,20),
                        '21st of October, 2021',
                        '2021-Oct-22',
                       '10-23-2021',
                       '20211024'])
dates

# - Any `DatetimeIndex` can be converted to a `PeriodIndex` with the `to_period()` function with the addition of a frequency code; here we use `'D'` to indicate daily frequency.

dates.to_period('D')

# - A `TimedeltaIndex` is created, for example, when a date is subtracted from another:

dates - dates[0]

# ## Regular Sequences: `pd.date_range()`

# - `pd.date_range()` for timestamsps, `pd.period_range()` for periods, and `pd.timedelta_range()` for time deltas. This is similar to Python's `range()` or `np.arange()`.

pd.date_range('2021-09-11','2021-10-21')

# - Alternatively, the date range can be specified not with a start and end point, but with a startpoint and a number of periods
# - The spacing can be modified by altering the `freq` argument, which defaults to `D`.

print(pd.date_range('2021-09-11',periods=10))
print(pd.date_range('2021-09-11', periods = 10, freq = 'H'))

# - To create regular sequencs of `Period` or `Timedelta` values, the very similar `pd.period_range()` and `pd.timedelta_range()` functions are useful. Here are some monthly periods:

pd.period_range('2021-09',periods = 10, freq='M')

# - A sequence of durations increasing by an hour:

pd.timedelta_range(0,periods=30, freq='H')


# Zehua Wang wangzeh@umich.edu

# ## Imports

import pandas as pd

# ## Question 0 - Topics in Pandas [25 points]

# ## Data Cleaning

# Create sample data
df = pd.DataFrame(
    {
        'col1': range(5),
        'col2': [6, 7, 8, 9, np.nan],
        'col3': [("red", "black")[i % 2] for i in range(5)],
        'col4': [("x", "y", "z")[i % 3] for i in range(5)],
        'col5': ["x", "y", "y", "x", "y"]
    }
)
df

# ### Duplicated Data
# - Find all values without duplication
# - Check if there is duplication using length comparison
# - return true if duplication exists

df['col3'].unique()
len(df['col3'].unique()) < len(df['col3'])

# ### Duplicated Data
# - Record duplication
# - subset: columns that need to remove duplication. Using all columns
#   if subset is None.
# - keep: Determine which duplicates to keep (if any), 'first' is default
#     - 'first': drop duplications except the first one
#     - 'last': drop duplications except the last one
#     - False: drop all duplications
# - inplace: return a copy (False, default) or drop duplicate (True)
# - ignore_index: return series label 0, 1, ..., n-1 if True, default is False

df.drop_duplicates()
df.drop_duplicates(subset=['col3'], keep='first', inplace=False)
df.drop_duplicates(subset=['col4', 'col5'], keep='last')

# ### Missing Data
# - Check if there is missing value
# - Delete missing value: pd.dropna
#     - axis: 0, delete by row; 1, drop by column
#     - how: any, delete if missing value exist; all, delete if 
#         all are missing values
#     - inplace: return a copy (False, default) or drop duplicate (True)    

df.isnull().any() # pd.notnull for selecting non-missing value
df.dropna(axis=0, how='any')

# ### Missing Data
# - Replcae missing value: pd.fillna
#     - value: the value filled up for missing value
#     - method: how to fill up the missing value
#         - 'backfill'/'bfill': using next valid observation
#         - 'pad'/'ffill': using previous valid observation
#         - None is by default
# - Generally, we could fill up the missing value with mean or median
#     for numeric data, and mode in categorical data.

df.fillna(method='ffill')
df.fillna(value=np.median(df[df['col2'].notnull()]['col2']))

# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [Filling in Missing Data](#Filling-in-Missing-Data) 
# + [Topic 2 Title](#Topic-2-Title)

# ## Topic Title
# Include a title slide with a short title for your content.
# Write your name in *bold* on your title slide. 

# ## Filling in Missing Data
#
#
# *Xinhe Wang*
#
# xinhew@umich.edu

# ## Fill in Missing Data
#
# - I will introduce some ways of using ```fillna()``` to fill in missing 
# data (```NaN``` values) in a DataFrame.
# - One of the most easiest ways is to drop the rows with missing values.
# - However, data is generally expensive and we do not want to lose all 
# the other columns of the row with missing data.
# - There are many ways to fill in the missing values:
#     - Treat the ```NaN``` value as a feature -> fill in with 0;
#     - Use statistics -> fill in with column mean/median/percentile/a
#     random value;
#     - Use the "neighbors" -> fill in with the last or next values;
#     - Prediction methods -> use regression/machine learning models to 
#     predict the missing value.

# ## Example Data
# - Here we generate a small example dataset with missing values.
#
# - Notice that if we want to indicate if the value in column "b" is larger
# than 0 in column "f", but for the missiing value in column "b", 
# ```df['b'] > 0``` returns ```False``` instead of a ```NaN``` value.
# Therefore, ```NaN``` values need to be delt with before further steps.

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5, 4),
                  columns=['a', 'b', 'c', 'd'])
df.iloc[2, 1] = np.nan
df.iloc[3:5, 0] = np.nan
df['e'] = [0, np.nan, 0, 0, 0]
df['f'] = df['b']  > 0
df

# ## Fill in with a scalar value
# - We can fill in ```NaN``` values with a designated value using 
# ```fillna()```.

df['e'].fillna(0)

df['e'].fillna("missing")

# ## Fill in with statistics (median, mean, ...)
# - One of the most commonly used techniques is to fill in missing values
# with column median or mean.
# - We show an instance of filling in missing values in column "b" with 
# column mean.

df['b'].fillna(df.mean()['b'])

# ## Fill in with forward or backward values
# - We can fill in with the missing values using its "neighber" using 
# ```fillna()```.
# - Can be used if the data is a time series.
# - When the ```method``` argument of ```fillna()``` is set as ```pad``` 
# or ```ffill```, values are filled forward; when ```method``` is set as
# ```bfill```or ```backfill```, values are filled backward.
# - The ```limit``` argument of ```fillna()``` sets the limit of number 
# of rows it is allowed to fill.

df['a'].fillna(method='pad', limit=1)

df['a'].fillna(method='bfill', limit=1)

# <p>This is a short tutorial about neat pandas idioms. <a href="https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#idioms">idioms</a> .
# From Xiatian Chen email:simoncxt@umich.edu</p>
# <h1>Idioms</h1>
# <h2>If-then and splitting:</h2>
# <pre><code>    -Clear idioms allow the code to be more readable and efficient  
#     -Always need to construct data under specific conditions, here are some examples.
# </code></pre>
# <p><code>df = pd.DataFrame(
#     {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
# )
# df.loc[df.AAA &gt;= 5, "BBB"] = -1</code>  </p>
# <pre><code>    -Can also apply if-then to multiple columns
# </code></pre>
# <p><code>df.loc[df.AAA &gt;= 5, ["BBB", "CCC"]] = 555</code>  </p>
# <pre><code>    -Can use numpy where() to apply if-then-else
# </code></pre>
# <p><code>df["logic"] = np.where(df["AAA"] &gt; 5, "high", "low")</code>  </p>
# <pre><code>    -Split the frame under condition
# </code></pre>
# <p><code>df[df.AAA &lt;= 5]
# df[df.AAA &gt; 5]</code> </p>
# <h2>Building criteria:</h2>
# <pre><code>    -When there is only 1-2 criterias, can be directly contained in df.loc  
#     -Can return a series or just modify the dataframe
# </code></pre>
# <p><code>df.loc[(df["BBB"] &lt; 25) &amp; (df["CCC"] &gt;= -40), "AAA"]
# df.loc[(df["BBB"] &gt; 25) | (df["CCC"] &gt;= 75), "AAA"] = 0.1</code>   </p>
# <pre><code>    -When there is a list of criteria, it can be done with a list of dynamically built criteria
# </code></pre>
# <p><code>Crit1 = df.AAA &lt;= 5.5
# Crit2 = df.BBB == 10.0
# Crit3 = df.CCC &gt; -40.0
# CritList = [Crit1, Crit2, Crit3]
# AllCrit = functools.reduce(lambda x, y: x &amp; y, CritList)
# df[AllCrit]</code> </p>

# In[ ]:


# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ### Author: Houyu Jiang
# ### Email: houyuj@umich.edu

# + [Topic: pd.diff()](#Topic:-pd.diff()) 
# + [Direction of the difference](#Direction-of-the-difference)
# + [Distance of difference](#Distance-of-difference)

# ## Topic: pd.diff()
#
# - ```pd.diff()``` is a pandas method that we could use to
# compute the difference between rows or columns in DataFrame.
# - We could import it through ```import pandas as pd```.
# - Suppose ```df``` is a pandas DataFrame, we could use 
# ```diff``` method to compute.

df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
                   'b': [1, 1, 2, 3, 5, 8],
                   'c': [1, 4, 9, 16, 25, 36]})
df.diff(axis=0)

# ## Direction of the difference
# - ```pd.diff()``` by default would calculate the 
# difference between different rows.
# - We could let it compute the difference between 
# previous columns by setting ```axis=1```

df.diff(axis=1)

# ## Distance of difference
# - ```pd.diff()``` by default would calculate the difference
# between this row/column and previous row/column
# - We could let it compute the difference between this row/column
# and the previous n row/column by setting ```periods=n```

df.diff(periods=3)


# In[ ]:


# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [Missing Data](#Missing-Data)

# ## Missing Data
# I will be looking at how pandas dataframes handle missing values.
# **Stefan Greenberg**
#
# sfgreen@umich.edu


# ## Imports
import numpy as np
import pandas as pd
# ## Detecting missing data
# - missing data includes `NaN`, `None`, and `NaT`
#     - can change settings so that `inf` and -`inf` count as missing
# - `.isna()` returns True wherever there is a missing value
# - `.notna()` returns True wherever there is not a missing value

# +
df = pd.DataFrame([[0.0, np.NaN, np.NaN, 3.0, 4.0, 5.0],
                   [0.0, 1.0, 4.0, np.NaN, 16.0, 25.0]], 
                 index=['n', 'n^2'])

df.append(df.isna())
# -

# ## Filling missing data
#
# - pandas makes it easy to replace missing values intelligently
# - the `.fillna()` method replaces all missing values with a given value
# - the `.interpolate()` method will use neighboring values to fill in gaps
# in data

# +
df_zeros = df.fillna(0)
df_interp = df.copy()

df_interp.loc['n'] = df_interp.loc['n']                     .interpolate(method='linear')
df_interp.interpolate(method='quadratic', axis=1, inplace=True)

df_zeros
#df_interp
# -

# ## Remove missing data with `.dropna()`
#
# - `.dropna()` will remove rows or columns that have missing values
# - set `axis` to determine whether to drop rows or columns
# - drop a row or column if it has any missing values or only if it has 
# entirely missing values by setting `how` to either *'any'* or *'all'*
# - set a minimum number of non-missing values required to drop row/column
# by setting `thresh`
# - specify what labels along other aixs to look at using `subset` i.e. 
# only drop a row if there is a missing value in a subset of the columns 
# or vise versa

# +
drop_cols   = df.dropna(axis=1)
drop_all    = df.dropna(how='all')
drop_thresh = df.dropna(thresh=5)
drop_subset = df.dropna(subset=[0, 1, 5])

print(df, '\n\n', 
      drop_cols.shape, drop_all.shape, drop_thresh.shape, drop_subset.shape)
# -
# ## Math operations with missing data
# - cumulative methods - `.cumsum()` and `.cumprod()` - by default will skip 
# over missing values
# - `.sum()` and `.prod()` treat missing values as identities
#     - `.sum()` treats missing values as zero
#     - `.prod()` treats missing values as one
#


# +
sumprod = df.append(
          df.sum()
            .to_frame()
            .transpose()
            .rename(index={0:'sum'}))

sumprod.append(
        df.prod()
          .to_frame()
          .transpose()
          .rename(index={0:'prod'}))
          
          
 #!/usr/bin/env python
# coding: utf-8

# Question 0
# 
# Name: Feng yuteng
# 
# Email:ytfeng@umich.edu

# In[6]:


import pandas as pd
import numpy as np


# ## Sparse data structures
# - overview
# - Sparse array 
# - SparseDtype
# - Sparse accessor

# ### Overview

# "So a matrix will be a sparse matrix if most of the elements of it is 0. Another definition is, a matrix with a maximum of 1/3 non-zero elements (roughly 30% of m x n) is known as sparse matrix. We use matrices in computers memory to do some operations in an efficient way."
# 
# References:" https://www.mvorganizing.org/what-is-sparse-matrix-in-data-structure-with-example/#What_is_sparse_matrix_in_data_structure_with_example"
# 

# ### Difference between a normal array and a sparse array

# - Sparse array(matrix) allocates spaces only for the non-default values.
# - Normal array(matrix) allocates spaces for all values. 
# - Therefore, sparse matrices are much cheaper to store since we only need to store certain entries of the matrix. 

# ## Sparse array

# In[12]:


arr = np.random.randn(10)
arr[2:5] = np.nan
arr[7:8] = np.nan
sparr = pd.arrays.SparseArray(arr)
sparr


# In[13]:


np.asarray(sparr)


# ## SparseDtype

# - The dtype of the non-sparse values
# 
# - The scalar fill value

# In[14]:


sparr.dtype


# In[16]:


pd.SparseDtype(np.dtype('datetime64[ns]'))
## default value will be used for filling missing value for that dtype


# ### Sparse accessor

# - ".sparse" provides attributes and methods that are specific to sparse data 

# In[23]:


s = pd.Series([0, 0, 1, 2], dtype="Sparse[int]")
s.sparse.density
## basically the mean of the series


# In[21]:


s.sparse.fill_value

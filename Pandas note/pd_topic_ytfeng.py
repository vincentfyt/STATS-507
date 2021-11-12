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


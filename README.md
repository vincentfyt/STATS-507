# STATS-507
STATS 507


For part A: We use Python and Pandas to read and append the demographic datasets keeping only columns containing the unique ids (SEQN), age (RIDAGEYR), race and ethnicity (RIDRETH3), education (DMDEDUC2), and marital status (DMDMARTL), along with the following variables related to the survey weighting: (RIDSTATR, SDMVPSU, SDMVSTRA, WTMEC2YR, WTINT2YR). Add an additional column identifying to which cohort each case belongs. Rename the columns with literate variable names using all lower case and convert each column to an appropriate type. Finally, save the resulting data frame to a serialized “round-trip” format of your choosing (e.g. pickle, feather, or parquet).



For part B: We repeat the same process as part we do in part A for the ohx dentition data. And we keep SEQN, OHDDESTS, tooth counts (OHXxxTC), and coronal cavities (OHXxxCTC).


For part C: We use the unique id to merge two datasets into one and figure out the number of cases appeared in both datasets.

[link](file:///C:/Users/fengy/Desktop/stats 507/hw6/Hw2,Question 3.ipynb)

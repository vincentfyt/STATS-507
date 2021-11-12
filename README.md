# STATS-507
STATS 507


For part A: We use Python and Pandas to read and append the demographic datasets keeping only columns containing the unique ids (SEQN), age (RIDAGEYR), race and ethnicity (RIDRETH3), education (DMDEDUC2), and marital status (DMDMARTL), along with the following variables related to the survey weighting: (RIDSTATR, SDMVPSU, SDMVSTRA, WTMEC2YR, WTINT2YR). Add an additional column identifying to which cohort each case belongs. Rename the columns with literate variable names using all lower case and convert each column to an appropriate type. Finally, save the resulting data frame to a serialized “round-trip” format of your choosing (e.g. pickle, feather, or parquet).



For part B: We repeat the same process as part we do in part A for the ohx dentition data. And we keep SEQN, OHDDESTS, tooth counts (OHXxxTC), and coronal cavities (OHXxxCTC).


For part C: We use the unique id to merge two datasets into one and figure out the number of cases appeared in both datasets.


**Update on PS4 Branch: Use the revised demographic data and the oral health data from PS2 to create a clean dataset with the following variables:**
- id (from SEQN)
- **- gender**
- age
- under_20 if age < 20
- college - with two levels:
- ‘some college/college graduate’ or
- ‘No college/<20’ where the latter category includes everyone under 20 years of age.
- exam_status (RIDSTATR)
- ohx_status - (OHDDESTS)
- Create a categorical variable in the data frame above named ohx with two levels “complete” for those with exam_status == 2 and ohx_status == 1 or “missing” when ohx_status is missing or corresponds to “partial/incomplete.”


-Remove rows from individuals with exam_status != 2 as this form of missingness is already accounted for in the survey weights. Report the number of subjects removed and the number remaining.


Construct a table with ohx (complete / missing) in columns and each of the following variables summarized in rows:
-age
-under_20
-gender
-college

[Local File Link] (https://github.com/vincentfyt/STATS-507/blob/f08f7374a8815787ae897d58569330bc82caea03/Hw2,Question%203.ipynb)


[Local File Link]（https://github.com/vincentfyt/STATS-507/blob/aac11729aa85c149832a81ec2edaf405133a7b74/homework4.ipynb)

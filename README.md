# Loan_Application_Predictor
Dream Housing Finance company deals in all home loans. They have a presence across all urban, semi-urban and rural areas. Customers first apply for a home loan after that company validates the customer’s eligibility for a loan. The company wants to automate the loan eligibility process (real-time) based on customer detail provided while filling out the online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others. To automate this process, they have given a problem to identify the customer segments, that are eligible for loan amounts so that they can specifically target these customers.

It is a classification problem where we have to predict whether a loan would be approved or not. In these kinds of problems, we have to predict discrete values based on a given set of independent variables.

***The links that I have used for the train and test splits of Data are -***
[Train.csv](https://raw.githubusercontent.com//dphi-official//Datasets//master//Loan_Data//loan_train.csv)
[Test.csv](https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv)
 
**I have run the code directly using the train and test URL instead of downloading. I find well-using URLs instead of downloading the data. You can do whatever you see well.**

**The set of libraries we are using for the Exploratory Data Analysis and the Machine-Learning libraries for the Classification is -**
```Python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.tree import  DecisionTreeClassifier


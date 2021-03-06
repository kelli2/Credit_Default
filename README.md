### Python Machine Learning: Kaggle Competition Home Credit Default August 2018

Author: Kelli (Davies) Roache

Please use this link to view the complete file with internal navigation links:
### http://nbviewer.jupyter.org/github/kelli2/Credit_Default/blob/master/Kaggle_HomeCredit_Davies.ipynb

### Summary
The goal of Home Credit Default Risk Kaggle competition was to predict which credit applicants are most likely to default on home loans. I entered as a solo contestant and the project here had an ROC AUC score of 0.79558 in the competition and ranked in the top 6.5% (#463 of 7198 teams), earning a bronze medal. The top score in the competition had an ROC AUC of 0.80570.

One challenge to the competition was working with large datafiles with lots of missing data. There were many features for over 300,000 loans. In the main application file, the median amount of missing data per loan was 30%. There were several supplemental files that often contained multiple rows for a given loan, and consequently the largest files had several million rows. Feature engineering was important because most features needed to be summarized for a given loan application and most features had very little predictive power on their own.

The approach was to first clean up the data, removing spurious values, normalizing some skewed values such as income, and converting categorical variables to binary variables to create a numeric only dataset. New feature generation was first done by considering intuitive relationships and summary statistics (mean, median, min, max, etc) as well as weighted values for time history values. Later, after all datasets were processed and combined, I generated more interaction features across all the data using all possible pairwise combinations.

I tried several data imputation methods and also worked with normalized and standardized versions of the data. These did not have strong benefits but in the end did provide a modest benefit to the final ROC AUC score. I used lightGBM models with hyperparameters identified via Bayesian Optimization and relied on cross fold validation to compare the models' performances on the different permutations of the train file. With the best models, I calculated default probablities for the test data. Prediction probabilities were then averaged to generate a stack/blended model that outperformed the individual models.


### Table of Contents
**[1. Introduction- Kaggle Competition Dataset from HomeCredit](#p1)<br>**
*Description of competition and provided datafiles.*

**[2. Open Files and Calculate Fraction Missing of Data](#p2)<br>**
*An intitial look at each file provided. Summaries of file size and breakdown of missing data.*

**[3. EDA and Processing of Application Train File](#p3)<br>**
*Deeper look at the main application file. Includes clean up of data, basic data imputation and effects on correlation with target, identification of colinear variables, feature engineering, and distributions of selected variables.*

&nbsp;&nbsp;&nbsp;&nbsp;[Initial Processing - Convert Binary Variables and Record Fraction Missing](#p31)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Analysis of Missing Data - Colinear Variables](#p32)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Missing Data Imputation - Comparison of Mean, Median and Mode fill methods](#p33)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Feature Selection - Identification and Removal Colinear Variables](#p34)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Class Imbalance of Loans](#p35)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Data Visualization of Selected Variables](#p36)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Conversion of Selected Multi-level Categorical Variables to Binary Variables](#p37)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Feature Engineering  From Main Application File](#p38)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Process Test Data](#p39)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Initial Look at Feature Importances and Comparison with Correlation Values](#p310)<br>

**[4. Processing and Feature Engineering of Supplemental Data Files:](#p4)<br>**
*In this section, I first confirm that the additional files have a substantial amount of data across the loans and then proceed with processing as was done for the application file in part 2.  This section makes heavy use of aggregate data to make new summary features. Many of the files have time-related data (such as payment histories) and so weighted metrics are also generated to take into account recency.*<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Most Loans Have Data in the Supplemental Files](#p41)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Bureau Data files](#p42)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Credit Card Balance](#p43)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Installment Payments](#p44)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Previous Application](#p45)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[POS Cash Balance](#p46)<br>

**[5. Merge All Data Files](#p5)<br>**
*Consolidate all processed datafiles into a single file, creating both imputated and non-imputated versions of the data for train and test.  Calculate feature importances for the non-imputated compiled train file.*

**[6. Generation of New Features via Feature interaction](#p6)<br>**
	*Unbiased approach to new features across all datasets with merged data file: generates all pairwise multiplication, division, addition, subtraction combinations. Comparison of new feature selection based on correlation versus feature importances. I observed slight improvement to ROC AUC with feature importances based selection of new interaction features.*

**[7. Data Normalization and Standardization](#p7)<br>**
*Generate normalized and standardized versions of the data which can be used with different machine learning algorithms, data imputation methods, and for generating new interaction variables.*

**[8. Comparison of Data Imputation Methods](#p8)<br>**
*Check if MICE, softimpute, mean, median, mode, and the “Multi-fill” (selection of mean, median and mode based on best correlation as used during processing in parts 3 and 4).  Unfilled dataset outperforms imputated versions.*

**[9. Interaction Feature Engineering to Generate Final Data Sets](#p9)<br>**
*Generate interaction features as done in part 6, but on the non-imputated and standardized versions of the data.*

**[10.  Bayesian Optimization of lightGBM Classifier Hyperparameters](#p10)<br>**
*Identification of best hyperparameters for several models that make use of different boosting algorithms.*

**[11. AUC ROC Values](#p11)<br>**
*Cross validation to assess which model on which training set provides the best score.*

**[12. Model Blending/Stacking and Results](#p12)<br>**
*Discussion of how final submission was generated and the results.*

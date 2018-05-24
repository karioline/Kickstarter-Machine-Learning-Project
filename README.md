# Kickstarter-Machine-Learning-Project
Final project code for CMPS 570 Machine Learning course. 

PROJECT DESCRIPTION
Big Idea:	How likely is it that a Kickstarter project will succeed?
Goal:		Build a model to predict whether a project will succeed.
Track 2:	Decision Tree and Random Forest Decision Tree	

Kickstarter is a crowdfunding platform that allows people to pledge money to projects to help fund them.
Data source: https://www.kaggle.com/kemical/kickstarter-projects/data 

DESCRIPTION OF DATA
Number of samples: 331,675 (~60% failed, ~40% succeed)
Preprocessing: Used one-hot-encoding

For LOO:
    Reduced number of samples used
        For dt_loo: 3,317 (1% of original data set)
        For rfdt_loo: 1,658 (0.5% of original data set)
    Reduced number of features
    
For the KNN and SVM bonus models:
    Used just main_category, backers, and usd_goal features
    Also used a categorical encoding for transforming the string values of main_category to numerical values ranging from 1 to 15. 


Check out  Final_Presentation.ppt in the Final_Project folder for analysis of results.

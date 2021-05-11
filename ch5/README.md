# Chapter 5: Classification

## logistic_regression.py
Uses logistic regression on the titanic dataset to predict survival. 
Prints accuracy report and plots ROC curve. Creates balloons_dt.png. 
    ORIGINAL: https://www.openml.org/d/40945 -> titanic.csv 
    CLEANED: titanic_data.csv

## softmax_regression.py
Makes predictions about 'prog' attribute of hsbdemo data, as described in 
textbook. Note: this program does not align well with the figures in the textbook.  
    ORIGINAL: IMT course website -> hsbdemo.csv

## decision_tree.py
Forms and visualizes a decision tree based on the UCI balloons data set. 
    ORIGINAL: https://archive.ics.uci.edu/ml/datasets/balloons -> balloons.csv
    
## random_forest.py
Builds a highly accurate random forest model over the UCI banking data set. 
    ORIGINAL: https://archive.ics.uci.edu/ml/datasets/bank+marketing -> bank.csv

## naive_bayes.py
Applies naive bayes to the banking data set. Compares multinomial
distribution assumption with Gaussian distribution assumption. 
    ORIGINAL: https://archive.ics.uci.edu/ml/datasets/bank+marketing -> bank.csv
    
## svm.py
Provides a comparison between SVM and logistic regression performance
on the skin data set. SVM outperforms logistic significantly, at the cost
of computation time. 
    ORIGINAL: https://archive.ics.uci.edu/ml/datasets/skin+segmentation -> Skin_NonSkin.txt
    REDUCED_SIZE: Skin_NonSkin_small.txt
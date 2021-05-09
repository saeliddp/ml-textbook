### Module Descriptions
logistic_regression.py - uses logistic regression on the titanic dataset to predict
    survival. Prints accuracy report and ROC curve.
    -> creates balloons_dt.png
    data: original https://www.openml.org/d/40945 -> titanic.csv
    After preprocessing to get rid of rows with missing data -> titanic_data.csv

softmax_regression.py - makes predictions about 'prog' attribute of hsbdemo data,
    as described in textbook. Note: this program does not align well with what is
    in the textbook. I'm not sure exactly what is necessary.
    data: IMT course website -> hsbdemo.csv

decision_tree.py - forms and visualizes a decision tree based on the UCI balloons
    data set
    data: https://archive.ics.uci.edu/ml/datasets/balloons -> balloons.csv
    
random_forest.py - builds a highly accurate random forest model over the UCI
    banking data set
    data: https://archive.ics.uci.edu/ml/datasets/bank+marketing -> bank.csv

naive_bayes.py - applies naive bayes to the banking data set. Compares multinomial
    distribution assumption with Gaussian distribution assumption.
    data: https://archive.ics.uci.edu/ml/datasets/bank+marketing -> bank.csv
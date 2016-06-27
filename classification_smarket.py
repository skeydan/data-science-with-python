from __future__ import division
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_selection import RFECV
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import discriminant_analysis
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

'''
*********************************************************************************************************************
                                 Functions
*********************************************************************************************************************
'''

def assess_classification_performance(model, X_train, y_train, X_test, y_test, short = False):
  
    accuracy_train = metrics.accuracy_score(y_train, model.predict(X_train))
    accuracy_test = metrics.accuracy_score(y_test, model.predict(X_test))
    print('accuracy (train/test): {} / {}\n'.format(accuracy_train, accuracy_test))
    
    if not short:
    
      # confusion matrix
      # rows: actual group
      # columns: predicted group
      print('Confusion_matrix (training data):')
      print(metrics.confusion_matrix(y_train, model.predict(X_train)))
      
      print('Confusion_matrix (test data):')
      print(metrics.confusion_matrix(y_test, model.predict(X_test)))

      # precision =  tp / (tp + fp)
      # recall = tp / (tp + fn) (= sensitivity)
      # F1 = 2 * (precision * recall) / (precision + recall)
      print('\nPrecision - recall (training data):')
      print(metrics.classification_report(y_train, model.predict(X_train)))
      
      print('\nPrecision - recall (test data):')
      print(metrics.classification_report(y_test, model.predict(X_test)))
    

# from https://github.com/rasbt/python-machine-learning-book
def plot_decision_regions(title, X, y, classifier, test_idx=None, resolution=0.02):

    plt.figure()
    plt.title(title)
    
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('cyan', 'red', 'lightgreen', 'gray', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
      
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')
    plt.show()    


# ROC curve
# specificity = tn / fn + fp
def print_ROC_curve(title, X, y, model):

    plt.figure()
    fpr_1, tpr_1, _ = metrics.roc_curve(y, model.predict_proba(X)[:,1])
    plt.plot(fpr_1, tpr_1, label = 'ROC curve for UP')
    fpr_0, tpr_0, _ = metrics.roc_curve(y, model.predict_proba(X)[:,0])
    plt.plot(fpr_0, tpr_0, label = 'ROC curve for DOWN')
    plt.legend(loc='upper left')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(title + ': ROC curves')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid(True)
    plt.show()


print '''
*********************************************************************************************************************
                                 Load dataset
*********************************************************************************************************************
'''

smarket = pd.read_csv('data/Smarket.csv').iloc[:,1:]


print '''
*********************************************************************************************************************
                                 Explore data
*********************************************************************************************************************
'''

print('\nSmarket dataset:')
print(smarket.head())
print('Columns: {}\n'.format(smarket.columns))
print('Index: {}\n'.format(smarket.index))
print('Correlation matrix:')
print(smarket.corr())


print '''
*********************************************************************************************************************
                                 Convert target variable to numerical
*********************************************************************************************************************
'''

smarket['dir_0_1'] = np.where(smarket['Direction'] == 'Up', 1, 0)
print('Correlation matrix:')
print(smarket.corr())


print '''
*********************************************************************************************************************
                                 Check for missing values and outliers
*********************************************************************************************************************
'''


print '''
*********************************************************************************************************************
                                 Split into train and test set
*********************************************************************************************************************
'''

x_columns = ['Lag1','Lag2','Lag3','Lag4','Lag5','Volume']
X_train = smarket[smarket['Year'] != 2005][x_columns].values
y_train = smarket[smarket['Year'] != 2005][['dir_0_1']].values[:,0]
X_test = smarket[smarket['Year'] == 2005][x_columns].values
y_test = smarket[smarket['Year'] == 2005][['dir_0_1']].values[:,0]



print '''
*********************************************************************************************************************
                                 Standardize
*********************************************************************************************************************
'''

# we scale the predictors coz Volume is on a different scale than the Lag<n> variables

# fit the scaler on training set and apply same to test set
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print scaler.mean_
print scaler.scale_

print X_train[:5,:]
print X_test[:5,:]


print '''


/******************************************************************************
*                              Logistic Regression                            *
******************************************************************************/

'''

logistic_model = linear_model.LogisticRegression()
logistic_model.fit(X_train, y_train)
print('Coefficients ({}): {}\n'.format(x_columns, logistic_model.coef_))

assess_classification_performance(logistic_model, X_train, y_train, X_test, y_test)      

# compare  with majority vote classifier
majority_vote_classifier_accuracy = max(y_test.mean(), 1 - y_test.mean())
print('Accuracy of majority vote classifier: {}\n'.format(majority_vote_classifier_accuracy))

print_ROC_curve('Logistic Regression', X_test, y_test, logistic_model)

# Feature ranking with recursive feature elimination and cross-validated selection of the best number of features
print("\nTrying sklearn.feature_selection.RFECV for feature selection:")
rfecv = RFECV(linear_model.LogisticRegression())
rfecv.fit(X_train, y_train)
print('Support: {}'.format(rfecv.support_))
print('Ranking: {}'.format(rfecv.ranking_))

print '''


/******************************************************************************
*              Logistic Regression using statsmodels                          *
******************************************************************************/

'''

# use statsmodels to get p values
sm_logistic = sm.GLM(y_train, X_train, sm.families.Binomial())
sm_results = sm_logistic.fit()
print sm_results.summary()


print '''


/******************************************************************************
*                 Logistic Regression using just Lag1                         *
******************************************************************************/

'''
X_train_lag1 = X_train[:,0:1]
X_test_lag1 = X_test[:,0:1]
logistic_model_lag1 = linear_model.LogisticRegression()
logistic_model_lag1.fit(X_train_lag1, y_train)
print('Coefficients ({}): {}\n'.format('Lag1', logistic_model_lag1.coef_))

assess_classification_performance(logistic_model_lag1,  X_train_lag1, y_train, X_test_lag1, y_test, short = True)

sm_logistic_lag1 = sm.GLM(y_train, X_train_lag1, sm.families.Binomial())
sm_results_lag1 = sm_logistic_lag1.fit()
print sm_results_lag1.summary()


print '''


/******************************************************************************
*                 Logistic Regression using Lag1 & Lag2                       *
******************************************************************************/

'''
X_train_lag12 = X_train[:,0:2]
X_test_lag12 = X_test[:,0:2]
logistic_model_lag12 = linear_model.LogisticRegression()
logistic_model_lag12.fit(X_train_lag12, y_train)
print('Coefficients ({}): {}\n'.format('Lag1, Lag2', logistic_model_lag12.coef_))

assess_classification_performance(logistic_model_lag12, X_train_lag12, y_train, X_test_lag12, y_test, short = True)

sm_logistic_lag12 = sm.GLM(y_train, X_train_lag12, sm.families.Binomial())
sm_results_lag12 = sm_logistic_lag12.fit()
print sm_results_lag12.summary()

plot_decision_regions('Logistic Regression', X_test_lag12, y_test, logistic_model_lag12)

print '''


/******************************************************************************
*               Logistic Regression using Lag1 & Volume                       *
******************************************************************************/

'''
X_train_lag1_vol = X_train[:,[0,5]]
X_test_lag1_vol = X_test[:,[0,5]]
logistic_model_lag1_vol = linear_model.LogisticRegression()
logistic_model_lag1_vol.fit(X_train_lag1_vol, y_train)
print('Coefficients ({}): {}\n'.format('Lag1, Volume', logistic_model_lag1_vol.coef_))


assess_classification_performance(logistic_model_lag1_vol,  X_train_lag1_vol, y_train, X_test_lag1_vol, y_test, short = True)

sm_logistic_lag1_vol = sm.GLM(y_train, X_train_lag1_vol, sm.families.Binomial())
sm_results_lag1_vol = sm_logistic_lag1_vol.fit()
print sm_results_lag1_vol.summary()


print '''


/******************************************************************************
*              Support Vector Classifier                                      *
******************************************************************************/

'''

svc_model = svm.SVC()
svc_model.fit(X_train, y_train)

assess_classification_performance(svc_model,  X_train, y_train, X_test, y_test, short = True)


print '''


/******************************************************************************
*              Support Vector Classifier using Lag1 & Lag2                    *
******************************************************************************/

'''

svc_model_lag12 = svm.SVC()
svc_model_lag12.fit(X_train_lag12, y_train)

assess_classification_performance(svc_model_lag12,  X_train_lag12, y_train, X_test_lag12, y_test, short = True)

plot_decision_regions('SVM', X_test_lag12, y_test, svc_model_lag12)



print '''


/******************************************************************************
*                            Decision tree                                    *
******************************************************************************/

'''

tree_model = tree.DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

assess_classification_performance(tree_model,  X_train, y_train, X_test, y_test, short = True)


print '''


/******************************************************************************
*                          Decision tree using Lag1 & Lag2                    *
******************************************************************************/

'''

tree_model_lag12 = tree.DecisionTreeClassifier()
tree_model_lag12.fit(X_train_lag12, y_train)

assess_classification_performance(tree_model_lag12,  X_train_lag12, y_train, X_test_lag12, y_test, short = True)

plot_decision_regions('Decision tree', X_test_lag12, y_test, tree_model_lag12)

print 'Node count: {}'.format(tree_model_lag12.tree_.node_count)

tree.export_graphviz(tree_model_lag12, out_file = 'tree.dot')
# dot -Tpng tree.dot -o tree.png

# constraint: max_leaf_nodes

for i in range(1,10):
  tree_model_lag12 = tree.DecisionTreeClassifier(max_depth = i)
  tree_model_lag12.fit(X_train_lag12, y_train)
  tree.export_graphviz(tree_model_lag12, out_file = 'tree_' + str(i) + '.dot')
  assess_classification_performance(tree_model_lag12,  X_train_lag12, y_train, X_test_lag12, y_test, short = True)
  #plot_decision_regions('Decision tree, max depth = ' + str(i), X_test_lag12, y_test, tree_model_lag12)


print '''
/******************************************************************************
*                            Random forest                                    *
******************************************************************************/

'''

forest_model = ensemble.RandomForestClassifier(n_estimators = 10000)
forest_model.fit(X_train, y_train)

assess_classification_performance(forest_model,  X_train, y_train, X_test, y_test, short = True)

# after 10000 runs, Lag1 seems more important than the rest
print('Feature importances: {}\n'.format(forest_model.feature_importances_))


print '''


/******************************************************************************
*                          Random forest using Lag1 & Lag2                    *
******************************************************************************/

'''

forest_model_lag12 = ensemble.RandomForestClassifier(n_estimators = 10000)
forest_model_lag12.fit(X_train_lag12, y_train)

assess_classification_performance(forest_model_lag12,  X_train_lag12, y_train, X_test_lag12, y_test, short = True)

plot_decision_regions('Random forest', X_test_lag12, y_test, forest_model_lag12)


print '''


/******************************************************************************
*                          AdaBoost using Lag1 & Lag2                    *
******************************************************************************/

'''
n_estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
for i in n_estimators:
  ada_model_lag12 = ensemble.AdaBoostClassifier(n_estimators = i)
  ada_model_lag12.fit(X_train_lag12, y_train)
  assess_classification_performance(ada_model_lag12,  X_train_lag12, y_train, X_test_lag12, y_test, short = True)
  plot_decision_regions('Ada Boost', X_test_lag12, y_test, ada_model_lag12)




print '''
/******************************************************************************
*                            Linear Discriminant Analysis                     *
******************************************************************************/

'''

# http://sebastianraschka.com/Articles/2014_python_lda.html#lda-via-scikit-learn
# http://scikit-learn.org/stable/modules/lda_qda.html#dimensionality-reduction-using-linear-discriminant-analysis

lda_model = discriminant_analysis.LinearDiscriminantAnalysis(n_components = 1, solver = 'eigen', store_covariance = True)
lda_model.fit(X_train, y_train)

print('Covariance matrix (shared by all classes): {}\n'.format(lda_model.covariance_))
print('Explained variance ratio: {}\n'.format(lda_model.explained_variance_ratio_))

assess_classification_performance(lda_model, X_train, y_train, X_test, y_test)



print '''
/******************************************************************************
*              Linear Discriminant Analysis using Lag1 & Lag2                 *
******************************************************************************/

'''

# http://sebastianraschka.com/Articles/2014_python_lda.html#lda-via-scikit-learn
# http://scikit-learn.org/stable/modules/lda_qda.html#dimensionality-reduction-using-linear-discriminant-analysis

lda_model_lag12 = discriminant_analysis.LinearDiscriminantAnalysis(n_components = 1, solver = 'eigen', store_covariance = True)
lda_model_lag12.fit(X_train_lag12, y_train)

print('Covariance matrix (shared by all classes): {}\n'.format(lda_model_lag12.covariance_))
print('Explained variance ratio: {}\n'.format(lda_model_lag12.explained_variance_ratio_))

assess_classification_performance(lda_model_lag12, X_train_lag12, y_train, X_test_lag12, y_test)
plot_decision_regions('LDA', X_test_lag12, y_test, lda_model_lag12)


print '''
/******************************************************************************
*    Linear Discriminant Analysis followed by Logistic Regression             *
******************************************************************************/

'''

X_train_ldatransform = lda_model.transform(X_train)
X_test_ldatransform = lda_model.transform(X_test)

print('Shape after LDA transform: {}'.format(X_train_ldatransform.shape))

logistic_model = linear_model.LogisticRegression()
logistic_model.fit(X_train_ldatransform, y_train)

assess_classification_performance(logistic_model, X_train_ldatransform, y_train, X_test_ldatransform, y_test)



print '''
/******************************************************************************
*                            Quadratic Discriminant Analysis                     *
******************************************************************************/

'''

qda_model = discriminant_analysis.QuadraticDiscriminantAnalysis(store_covariances = True)
qda_model.fit(X_train, y_train)

print('Covariance matrices: {}\n'.format(qda_model.covariances_))

assess_classification_performance(qda_model, X_train, y_train, X_test, y_test)



print '''
/******************************************************************************
*                 Quadratic Discriminant Analysis using Lag1 & Lag2           *
******************************************************************************/

'''

qda_model_lag12 = discriminant_analysis.QuadraticDiscriminantAnalysis(store_covariances = True)
qda_model_lag12.fit(X_train_lag12, y_train)

print('Covariance matrices: {}\n'.format(qda_model_lag12.covariances_))

assess_classification_performance(qda_model_lag12, X_train_lag12, y_train, X_test_lag12, y_test)
plot_decision_regions('LDA', X_test_lag12, y_test, qda_model_lag12)






from __future__ import division
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import linear_model 
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import neighbors
import seaborn as sns
import matplotlib.pyplot as plt


'''
*********************************************************************************************************************
                                 Functions
*********************************************************************************************************************
'''

def eval_performance(model, X_train, y_train, X_test, y_test):
  
  print("\nEvaluating model: %s" % model)
  
  predictions_train = model.predict(X_train)
  residuals_train = predictions_train - y_train
  predictions_test = model.predict(X_test)
  residuals_test = predictions_test - y_test
  
  # RSS 
  RSS_train = (residuals_train ** 2).sum()
  RSS_test = (residuals_test ** 2).sum()
  print("Residual sum of squares (train): %d" % RSS_train)
  print("Residual sum of squares (test): %d" % RSS_test)
  
  # Mean squared error
  print('MSE (train): %.2f' % mean_squared_error(y_train, predictions_train))
  print('MSE (test): %.2f' % mean_squared_error(y_test, predictions_test))
  
  # R^2 = 1 - (residual sum of squares / total sum of squares) = 1 - MSE/Var(y)
  # R_squared_test = 1 - RSS_test / ((y_test - y_test.mean()) **2).sum()
  print('R^2 (train): %.2f' % model.score(X_train, y_train))
  print('R^2 (test): %.2f\n' % model.score(X_test, y_test))
  
  if print_residual_plots:
    # residual plot for training data and test data
    plt.figure()
    plt.scatter(predictions_train, residuals_train, c='cyan', marker='s', label='training data')
    plt.scatter(predictions_test, residuals_test, c='red', marker='o', label='test data')
    #plt.hlines(y=0, xmin=-10, xmax=50)
    plt.legend()
    plt.xlabel('predicted values')
    plt.ylabel('residuals')
    plt.show()
  

def coefficient_path(model, alphas, X, y):
  model = model
  coefs = []
  for a in alphas:
    model.set_params(alpha=a)
    model.fit(X, y)
    coefs.append(model.coef_)
  plt.figure()
  ax = plt.gca()
  ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
  ax.plot(alphas, coefs)
  ax.set_xscale('log')
  ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
  plt.xlabel('alpha')
  plt.ylabel('weights')
  plt.title('Coefficients as a function of the regularization')
  plt.axis('tight')
  plt.show()


def plot_validation_curve(estimator, X, y, param_name, param_range):
  
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

    plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

    plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0., 1.0])
    plt.tight_layout()
    plt.show()


print '''
*********************************************************************************************************************
                                 Load dataset
*********************************************************************************************************************
'''


diabetes = load_diabetes()


print '''
*********************************************************************************************************************
                                 Explore data
*********************************************************************************************************************
'''

print_residual_plots = False

print diabetes.keys()

print(diabetes.data.shape)

X = diabetes.data
y = diabetes.target

column_names=['age', 'sex', 'bmi', 'map', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']

# for purpose of exploration, build one DataFrame of X and y data
df = pd.DataFrame(X, columns = column_names)
df['target'] = pd.Series(y)

print(df.head())

print df.describe()

# correlations among input variables
print df.corr()
#scatter_matrix(df, diagonal='kde')

'''
plt.figure()
coefs = np.corrcoef(df.values.T)
sns.set(style='whitegrid')
hm = sns.heatmap(coefs, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=df.columns, xticklabels=df.columns) 
plt.show()
sns.reset_orig()
'''


print '''
*********************************************************************************************************************
                                 Split into training and test set
*********************************************************************************************************************
'''

X_train, X_test, y_train, y_test = cross_validation.train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)



print '''
*********************************************************************************************************************
                                 Linear regression
*********************************************************************************************************************
'''

lreg_model = linear_model.LinearRegression()
lreg_model.fit(X_train, y_train)

print('Coefficients: \n', lreg_model.coef_)

eval_performance(lreg_model, X_train, y_train, X_test, y_test)




print '''
*********************************************************************************************************************
                                 Lasso regression
*********************************************************************************************************************
'''

# minimize RSS + alpha * ||w||1

alphas = np.logspace(-5, 2, 50)

lasso_model_cv = linear_model.LassoCV(alphas=alphas, cv=20)
lasso_model_cv.fit(X_train, y_train)  

print('Coefficients: \n', lasso_model_cv.coef_)
eval_performance(lasso_model_cv, X_train, y_train, X_test, y_test)
print('Best Lasso alpha: {}\n'.format(lasso_model_cv.alpha_)) 

coefficient_path(linear_model.Lasso(), alphas, X_train, y_train)
#plot_validation_curve(linear_model.Lasso(), X_train, y_train, 'alpha', alphas)


lasso_model = linear_model.Lasso(alpha=0.9)
lasso_model.fit(X_train, y_train)  
print('Coefficients: \n', lasso_model.coef_)
eval_performance(lasso_model, X_train, y_train, X_test, y_test)

lasso_model = linear_model.Lasso(alpha=0.6)
lasso_model.fit(X_train, y_train)  
print('Coefficients: \n', lasso_model.coef_)
eval_performance(lasso_model, X_train, y_train, X_test, y_test)


print '''
*********************************************************************************************************************
                                k Nearest Neighbors Regression 
*********************************************************************************************************************
'''

n_neighbors = [3,5,10,20]
weights = ['uniform', 'distance']

for n in n_neighbors:
  for w in weights:
    
    print('knn (n = {}, weights = {})\n'.format(n,w))
    knn_model = neighbors.KNeighborsRegressor(n, weights = w)
    knn_model.fit(X_train, y_train)
    eval_performance(knn_model, X_train, y_train, X_test, y_test)





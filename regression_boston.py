from __future__ import division
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import linear_model 
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
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


boston = load_boston()


print '''
*********************************************************************************************************************
                                 Explore data
*********************************************************************************************************************
'''

print_residual_plots = True
print_all_scattermatrices = False

print boston.keys()
# ['data', 'feature_names', 'DESCR', 'target']

print boston.feature_names
# array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='|S7')

print boston.DESCR
# CRIM     per capita crime rate by town\n 
# ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n  
# INDUS    proportion of non-retail business acres per town\n     
# CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n  
# NOX      nitric oxides concentration (parts per 10 million)\n
# RM       average number of rooms per dwelling\n
# AGE      proportion of owner-occupied units built prior to 1940\n 
# DIS      weighted distances to five Boston employment centres\n 
# RAD      index of accessibility to radial highways\n
# TAX      full-value property-tax rate per $10,000\n
# PTRATIO  pupil-teacher ratio by town\n 
# B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n  
# LSTAT    % lower status of the population\n
# MEDV     Median value of owner-occupied homes in $1000's\n\n  

print(boston.data.shape)

X = boston.data
y = boston.target

# for purpose of exploration, build one DataFrame of X and y data
df = pd.DataFrame(X, columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
df['MEDV'] = pd.Series(y)

print(df.head())

# check for categorical values
print df.dtypes

print df.describe()

# visualize distributions
df.plot.box(subplots=True, sharex=False, sharey = False, layout=(4,4))
df.plot.kde(subplots=True, sharex=False, sharey = False, layout=(4,4))

# correlations among input variables
print df.corr()
if print_all_scattermatrices: scatter_matrix(df, diagonal='kde')

# only look at variables correlated > abs(0.5)
correlations_gt_05 = df.corr().where(abs(df.corr()) > 0.5, np.NaN)
print correlations_gt_05
cols_gt_05 = ['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO', 'LSTAT','MEDV']
if print_all_scattermatrices: scatter_matrix(df[cols_gt_05], diagonal='kde')

# variables correlated with MEDV > abs(0.5)
cor_medv_gt_05 = df.corr().loc['MEDV'][abs(df.corr().loc['MEDV']) > 0.5]
cols_corr_medv = list(cor_medv_gt_05.index.values)

scatter_matrix(df[cols_corr_medv], diagonal='kde')

# heatmap for variables correlated > abs(0.5)
plt.figure()
coefs = np.corrcoef(df[cols_gt_05].values.T)
sns.set(style='whitegrid')
hm = sns.heatmap(coefs, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols_gt_05, xticklabels=cols_gt_05) 
plt.show()
sns.reset_orig()

print '''
*********************************************************************************************************************
                                 Check for missing values and outliers
*********************************************************************************************************************
'''
# https://www.oreilly.com/learning/handling-missing-data
# http://scikit-learn.org/stable/auto_examples/missing_values.html#example-missing-values-py
# null_indices = np.where(X_df.isnull())
# not any(map(lambda x: x==0, null_indices))
print("No missing values found: %s" % np.all(df[df.notnull()] == df))


# check outliers


print '''
*********************************************************************************************************************
                                 Split into training and test set
*********************************************************************************************************************
'''

X_train, X_test, y_train, y_test = cross_validation.train_test_split(boston.data, boston.target, test_size=0.2, random_state=0)


print '''
*********************************************************************************************************************
                                 Standardize / Normalize
*********************************************************************************************************************
'''

# fit the scaler on training set and apply same to test set
#scaler = preprocessing.StandardScaler().fit(X_train)
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#print scaler.mean_
#print scaler.scale_
print scaler.data_min_
print scaler.data_max_

print X_train[:5,:]
print X_test[:5,:]


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
                                 Ridge regression
*********************************************************************************************************************
'''

# minimize RSS + alpha * ||w||2 ** 2

alphas = np.logspace(-5, 2, 50)

ridge_model_cv = linear_model.RidgeCV(alphas=alphas, store_cv_values=True)
ridge_model_cv.fit(X_train, y_train)  
print('Coefficients: \n', ridge_model_cv.coef_)

eval_performance(ridge_model_cv, X_train, y_train, X_test, y_test)
print('Best ridge alpha: {}\n'.format(ridge_model_cv.alpha_)) 
#print ridge_model_cv.cv_values_

coefficient_path(linear_model.Ridge(), alphas, X_train, y_train)
plot_validation_curve(linear_model.Ridge(), X_train, y_train, 'alpha', alphas)


print '''
*********************************************************************************************************************
                                 Lasso regression
*********************************************************************************************************************
'''

# minimize RSS + alpha * ||w||1

lasso_model_cv = linear_model.LassoCV(alphas=alphas)
lasso_model_cv.fit(X_train, y_train)  

print('Coefficients: \n', lasso_model_cv.coef_)
eval_performance(lasso_model_cv, X_train, y_train, X_test, y_test)
print('Best Lasso alpha: {}\n'.format(lasso_model_cv.alpha_)) 

coefficient_path(linear_model.Lasso(), alphas, X_train, y_train)
plot_validation_curve(linear_model.Lasso(), X_train, y_train, 'alpha', alphas)


print '''
*********************************************************************************************************************
                                 Polynomial features, using just LSTAT
*********************************************************************************************************************
'''

#poly_cols = ['RM', 'PTRATIO', 'LSTAT']
#X_train_poly = X_train[:,[5,10,12]]

X_train_lstat = X_train[:,12:13]

X_train_lstat_poly_1 = X_train_lstat
poly_2 = PolynomialFeatures(degree=2, include_bias=False)
poly_3 = PolynomialFeatures(degree=3, include_bias=False)
poly_4 = PolynomialFeatures(degree=4, include_bias=False)

X_train_lstat_poly_2 = poly_2.fit_transform(X_train_lstat)
X_train_lstat_poly_3 = poly_3.fit_transform(X_train_lstat)
X_train_lstat_poly_4 = poly_4.fit_transform(X_train_lstat)

print X_train_lstat_poly_2[:3,:]
print poly_2.n_output_features_
print poly_2.powers_
print X_train_lstat_poly_3[:3,:]
print poly_3.n_output_features_
print poly_3.powers_
print X_train_lstat_poly_4[:3,:]
print poly_4.n_output_features_
print poly_4.powers_

X_range = np.arange(X_test[:,12:13].min(), X_test[:,12:13].max(), 0.01)[:, np.newaxis]

lreg_model = linear_model.LinearRegression()
lreg_model.fit(X_train_lstat_poly_1, y_train)
print('Coefficients: \n', lreg_model.coef_)
poly_1_prediction_curve = lreg_model.predict(X_range)
eval_performance(lreg_model, X_train_lstat_poly_1, y_train, X_test[:,12:13], y_test)
r2_score_1 = r2_score(y_test, lreg_model.predict(X_test[:,12:13]))

lreg_model.fit(X_train_lstat_poly_2, y_train)
print('Coefficients: \n', lreg_model.coef_)
poly_2_prediction_curve = lreg_model.predict(poly_2.transform(X_range))
eval_performance(lreg_model, X_train_lstat_poly_2, y_train, poly_2.transform(X_test[:,12:13]), y_test)
r2_score_2 = r2_score(y_test, lreg_model.predict(poly_2.transform(X_test[:,12:13])))

lreg_model.fit(X_train_lstat_poly_3, y_train)
print('Coefficients: \n', lreg_model.coef_)
poly_3_prediction_curve = lreg_model.predict(poly_3.transform(X_range))
eval_performance(lreg_model, X_train_lstat_poly_3, y_train, poly_3.transform(X_test[:,12:13]), y_test)
r2_score_3 = r2_score(y_test, lreg_model.predict(poly_3.transform(X_test[:,12:13])))

lreg_model.fit(X_train_lstat_poly_4, y_train)
print('Coefficients: \n', lreg_model.coef_)
poly_4_prediction_curve = lreg_model.predict(poly_4.transform(X_range))
eval_performance(lreg_model, X_train_lstat_poly_4, y_train, poly_4.transform(X_test[:,12:13]), y_test)
r2_score_4 = r2_score(y_test, lreg_model.predict(poly_4.transform(X_test[:,12:13])))

plt.figure()
plt.scatter(X_test[:,12:13], y_test, label='test points', color='lightgray')
plt.plot(X_range, poly_1_prediction_curve, label='1d, R^2 = %.2f' % r2_score_1, color='k', linestyle='-')
plt.plot(X_range, poly_2_prediction_curve, label='2d, R^2 = %.2f' % r2_score_2, color='b', linestyle='--')
plt.plot(X_range, poly_3_prediction_curve, label='3d, R^2 = %.2f' % r2_score_3, color='c', linestyle='-')
plt.plot(X_range, poly_4_prediction_curve, label='4d, R^2 = %.2f' % r2_score_4, color='r', linestyle='--')
plt.xlabel ('LSTAT')
plt.ylabel('MEDV')
plt.legend()
plt.show()


print '''
*********************************************************************************************************************
                                 High-degree polynomials, using just LSTAT
*********************************************************************************************************************
'''

poly_8 = PolynomialFeatures(degree=8, include_bias=False)
poly_12 = PolynomialFeatures(degree=12, include_bias=False)
poly_16 = PolynomialFeatures(degree=16, include_bias=False)

X_train_lstat_poly_8 = poly_8.fit_transform(X_train_lstat)
X_train_lstat_poly_12 = poly_12.fit_transform(X_train_lstat)
X_train_lstat_poly_16 = poly_16.fit_transform(X_train_lstat)

lreg_model = linear_model.LinearRegression()
lreg_model.fit(X_train_lstat_poly_8, y_train)
print('Coefficients: \n', lreg_model.coef_)
poly_8_prediction_curve = lreg_model.predict(poly_8.transform(X_range))
poly_8_training_predictions = lreg_model.predict(X_train_lstat_poly_8)
eval_performance(lreg_model, X_train_lstat_poly_8, y_train, poly_8.transform(X_test[:,12:13]), y_test)
r2_score_8_train = r2_score(y_train, lreg_model.predict(X_train_lstat_poly_8))
r2_score_8 = r2_score(y_test, lreg_model.predict(poly_8.transform(X_test[:,12:13])))

lreg_model.fit(X_train_lstat_poly_12, y_train)
print('Coefficients: \n', lreg_model.coef_)
poly_12_prediction_curve = lreg_model.predict(poly_12.transform(X_range))
poly_12_training_predictions = lreg_model.predict(X_train_lstat_poly_12)
eval_performance(lreg_model, X_train_lstat_poly_12, y_train, poly_12.transform(X_test[:,12:13]), y_test)
r2_score_12_train = r2_score(y_train, lreg_model.predict(X_train_lstat_poly_12))
r2_score_12 = r2_score(y_test, lreg_model.predict(poly_12.transform(X_test[:,12:13])))

lreg_model.fit(X_train_lstat_poly_16, y_train)
print('Coefficients: \n', lreg_model.coef_)
poly_16_prediction_curve = lreg_model.predict(poly_16.transform(X_range))
poly_16_training_predictions = lreg_model.predict(X_train_lstat_poly_16)
eval_performance(lreg_model, X_train_lstat_poly_16, y_train, poly_16.transform(X_test[:,12:13]), y_test)
r2_score_16_train = r2_score(y_train, lreg_model.predict(X_train_lstat_poly_16))
r2_score_16 = r2_score(y_test, lreg_model.predict(poly_16.transform(X_test[:,12:13])))


plt.figure()
plt.scatter(X_train_lstat, y_train, label='training points', color='lightgray')
plt.plot(np.sort(X_train_lstat[:,0]), np.sort(poly_8_training_predictions), label='8d, R^2 = %.2f' % r2_score_8_train, color='k', linestyle='-')
plt.plot(np.sort(X_train_lstat[:,0]), np.sort(poly_12_training_predictions), label='12d, R^2 = %.2f' % r2_score_12_train, color='b', linestyle='--')
plt.plot(np.sort(X_train_lstat[:,0]), np.sort(poly_16_training_predictions), label='16d, R^2 = %.2f' % r2_score_16_train, color='c', linestyle='-')
plt.xlabel ('LSTAT')
plt.ylabel('MEDV')
plt.ylim(bottom=0,top=60)
plt.legend()
plt.show()


plt.figure()
plt.scatter(X_test[:,12:13], y_test, label='test points', color='lightgray')
plt.plot(X_range, poly_8_prediction_curve, label='8d, R^2 = %.2f' % r2_score_8, color='k', linestyle='-')
plt.plot(X_range, poly_12_prediction_curve, label='12d, R^2 = %.2f' % r2_score_12, color='b', linestyle='--')
plt.plot(X_range, poly_16_prediction_curve, label='16d, R^2 = %.2f' % r2_score_16, color='c', linestyle='-')
plt.xlabel ('LSTAT')
plt.ylabel('MEDV')
plt.ylim(bottom=0,top=60)
plt.legend()
plt.show()



print '''
*********************************************************************************************************************
                                 16 deg. polynomial of LSTAT, with regularization
*********************************************************************************************************************
'''
print '''***    Standard linear regression   ***'''

poly_16 = PolynomialFeatures(degree=16, include_bias=False)
X_train_lstat_poly_16 = poly_16.fit_transform(X_train_lstat)

lreg_model = linear_model.LinearRegression()
lreg_model.fit(X_train_lstat_poly_16, y_train)
print('Coefficients: \n', lreg_model.coef_)

eval_performance(lreg_model, X_train_lstat_poly_16, y_train, poly_16.transform(X_test[:,12:13]), y_test)


print '''***    Ridge regression   ***'''

alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 10.0, 100.0]
#alphas = np.logspace(-7, 1, 10)

ridge_model_cv = linear_model.RidgeCV(alphas = alphas, store_cv_values=True)
ridge_model_cv.fit(X_train_lstat_poly_16, y_train)  
print('Coefficients: \n', ridge_model_cv.coef_)
eval_performance(ridge_model_cv, X_train_lstat_poly_16, y_train, poly_16.transform(X_test[:,12:13]), y_test)
print('Best ridge alpha: {}\n'.format(ridge_model_cv.alpha_)) 
#print ridge_model_cv.cv_values_

coefficient_path(linear_model.Ridge(), alphas, X_train_lstat_poly_16, y_train)

plot_validation_curve(linear_model.Ridge(), X_train_lstat_poly_16, y_train, 'alpha', alphas)


print '''***    Lasso regression   ***'''

lasso_model_cv = linear_model.LassoCV(alphas=alphas, max_iter=100000)
lasso_model_cv.fit(X_train_lstat_poly_16, y_train)  
print('Coefficients: \n', lasso_model_cv.coef_)
eval_performance(lasso_model_cv, X_train_lstat_poly_16, y_train, poly_16.transform(X_test[:,12:13]), y_test)
print('Best Lasso alpha: {}\n'.format(lasso_model_cv.alpha_)) 

coefficient_path(linear_model.Lasso(), alphas, X_train_lstat_poly_16, y_train)

plot_validation_curve(linear_model.Lasso(), X_train_lstat_poly_16, y_train, 'alpha', alphas)



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




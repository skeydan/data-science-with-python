from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import feature_selection
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import discriminant_analysis
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
    

# ROC curve
# specificity = tn / fn + fp
def print_ROC_curve(title, X, y, model):
  
    if len(categories) != 2: return

    plt.figure()
    fpr_1, tpr_1, _ = metrics.roc_curve(y, model.predict_proba(X)[:,1])
    plt.plot(fpr_1, tpr_1, label = 'ROC curve')
    plt.legend(loc='lower right')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(title + ': ROC curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid(True)
    plt.show()




print '''
*********************************************************************************************************************
                                 Load dataset
*********************************************************************************************************************
'''

all_categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
#categories = all_categories
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories = categories)
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories = categories)

classes = newsgroups_train.target_names
print('Class names: {}\n'.format(classes))


print '''
*********************************************************************************************************************
                                 Explore data
*********************************************************************************************************************
'''

print newsgroups_train.filenames.shape
print newsgroups_train.target.shape
print newsgroups_train.filenames[:5]
print newsgroups_train.target[:5]


print '''
*********************************************************************************************************************
                                 Transform for tf-idf
*********************************************************************************************************************
'''

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
print X_train.shape
y_train = newsgroups_train.target

X_test = vectorizer.transform(newsgroups_test.data)
print X_test.shape
y_test = newsgroups_test.target

tf_idfs = vectorizer.vocabulary_
tf_idfs_sorted = sorted(tf_idfs.iteritems(), key=tf_idfs.get)
tf_idfs_sorted[:20]


print '''
/******************************************************************************
*                              Logistic Regression                            *
******************************************************************************/
'''

logistic_model = linear_model.LogisticRegression()
logistic_model.fit(X_train, y_train)
print('Coefficients shape: {}\n'.format(logistic_model.coef_.shape))

print('Class names (for classes 0-3): {}\n'.format(classes))

assess_classification_performance(logistic_model, X_train, y_train, X_test, y_test, short = True)      
print_ROC_curve('Logistic Regression',X_test, y_test, logistic_model)


print '''
/******************************************************************************
*                   L2 regularized Logistic Regression                        *
******************************************************************************/
'''

logistic_model_l2 = linear_model.LogisticRegressionCV(Cs=[1e2,1e3,1e4,1e5,1e6,1e7], n_jobs=4)
logistic_model_l2.fit(X_train, y_train)
print('Coefficients shape: {}\n'.format(logistic_model_l2.coef_.shape))

print('Regularization coefficients used in cross validation: {}\n'.format(logistic_model_l2.Cs_))
print('Best regularization coefficient per class: {}\n'.format(logistic_model_l2.C_))

print('Class names (for classes 0-3): {}\n'.format(classes))

assess_classification_performance(logistic_model_l2, X_train, y_train, X_test, y_test, short = True)      
print_ROC_curve('L2 Regularized Logistic Regression', X_test, y_test, logistic_model_l2)  



print '''
/******************************************************************************
*                   L1 regularized Logistic Regression                        *
******************************************************************************/
'''

logistic_model_l1 = linear_model.LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=[1e2,1e3,1e4,1e5,1e6,1e7], n_jobs=4)
logistic_model_l1.fit(X_train, y_train)
print('Coefficients shape: {}\n'.format(logistic_model_l1.coef_.shape))

print('Regularization coefficients used in cross validation: {}\n'.format(logistic_model_l1.Cs_))
print('Best regularization coefficient per class: {}\n'.format(logistic_model_l1.C_))

print('Number of nonzero coefficients (class 0): {} of {} overall\n'.format(logistic_model_l1.coef_.shape[1] - logistic_model_l1.coef_[0][logistic_model_l1.coef_[0] == 0].size, logistic_model_l1.coef_.shape[1]))

print('Class names (for classes 0-3): {}\n'.format(classes))

assess_classification_performance(logistic_model_l1, X_train, y_train, X_test, y_test, short = True)      
print_ROC_curve('L1 Regularized Logistic Regression', X_test, y_test, logistic_model_l1)      



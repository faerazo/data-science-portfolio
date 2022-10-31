#!/usr/bin/env python
# coding: utf-8

# # Premium Product Predictor

# Build a model which can predict whether a product is premium or non-premium.
# 
# Disclaimer:
# Due to confidentiality reasons, it is not possible to present or divulge all aspects of the dataset used in this project.

# ## Data Preparation

# In[8]:


# libraries
import pyarrow.parquet as pq
import numpy as np


# In[9]:


# reading data
table = pq.read_table('/.../Premium Product Predictor/dataset.parquet')
df = table.to_pandas()


# In[10]:


# checking columns names
sorted(df.columns)


# In[11]:


column_names_reordered = [
    'column_33', 'column_16', 'column_17', 'column_19', 'column_22', 'column_23', 
    'column_24', 'column_25', 'column_27', 'column_35', 'column_36', 'column_40', 
    'column_41', 'column_45', 'column_46', 'column_48', 'column_50', 'column_52', 
    'column_56', 'column_58', 'column_60', 'column_63', 'column_64', 'column_65', 
    'column_68', 'column_71', 'column_72', 'column_73', 'column_74', 'column_75',
    'column_79', 'column_80', 'column_81', 'column_82', 'column_94', 'column_98', 
    'column_99']

df = df[column_names_reordered]


# In[12]:


# checking for null values
df.isna().sum()


# In[13]:


# checking for duplicated values
df.duplicated().sum()


# In[14]:


# getting unique values for each variable
for col in df:
    print(col, '=', df[col].unique())


# In[15]:


# changing data type from float to int
df['column_33'] = df['column_33'].astype(np.int64)
df['column_72'] = df['column_72'].astype(np.int64)
df['column_35'] = df['column_35'].astype(np.int64)
df['column_27'] = df['column_27'].astype(np.int64)
df['column_56'] = df['column_56'].astype(np.int64)
df['column_73'] = df['column_73'].astype(np.int64)
df['column_58'] = df['column_58'].astype(np.int64)


# In[16]:


# drop column_35 and column_72
# column_35 generates data leakage
# column_72 is not relevant according to the domain experts
df = df.drop(['column_35', 'column_72'], axis=1)


# In[18]:


# saving to a CSV file
df.to_csv('/.../Premium Product Predictor/dataset.csv', index=False)


# ## Data Exploration

# In[1]:


# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve,    log_loss, precision_recall_curve, auc
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold


# In[2]:


# reading data
df = pd.read_csv('/.../Premium Product Predictor/dataset.csv')


# In[3]:


# imbalanced classes
sns.countplot(x=df['column_33']).bar_label(sns.countplot(x=df['column_33']).containers[0])
plt.show()


# In[4]:


print('Descriptive Statistics')
df.describe().T


# In[5]:


print('Skewness')
print(df.skew(axis=0))


# In[6]:


print('Kurtosis')
df.kurtosis()


# In[6]:


# visualize distributions of all variables
cols = df.columns
fig, ax = plt.subplots(nrows=12, ncols=3, figsize=(25, 50))
ax = ax.ravel()

for col, i in zip(cols, ax):
    sns.histplot(data=df[col], ax=i)


# In[7]:


# visualize distributions of all variables by class
fig, ax = plt.subplots(nrows=12, ncols=3, figsize=(25, 50))
ax = ax.ravel()

for col, i in zip(cols, ax):
    sns.histplot(data=df, x=col, hue='column_33', ax=i)


# In[8]:


# correlation matrix
correlation_matrix = df.corr(method='pearson').abs()
correlation_greater_than_0_5 = correlation_matrix[correlation_matrix > 0.5]
mask = np.triu(np.ones_like(correlation_matrix.corr(method='pearson')))
sns.set(rc={'figure.figsize': (30, 30)})
sns.heatmap(correlation_greater_than_0_5, annot=True, cmap='coolwarm', mask=mask)


# In[9]:


def drop_pairs(dataframe):
    """
    Return the pairs that can be dropped from the correlation matrix.
    """
    pairs_to_drop = set()
    columns = dataframe.columns
    for j in range(0, dataframe.shape[1]):
        for k in range(0, j + 1):
            pairs_to_drop.add((columns[j], columns[k]))
    return pairs_to_drop


def sort_correlations(dataframe, n=10):
    """
    Return the top-10 pairs with the highest correlation values (without duplicates pairs).
    """
    corr_pairs = df.corr().abs().unstack()
    pairs_to_drop = drop_pairs(dataframe)
    corr_pairs = corr_pairs.drop(labels=pairs_to_drop).sort_values(ascending=False)
    return corr_pairs[0:n]


# In[10]:


# top 10 correlations
sort_correlations(df)


# In[11]:


# boxplot
cols = df.columns
fig, ax = plt.subplots(nrows=12, ncols=3, figsize=(25, 50))
ax = ax.ravel()

for col, i in zip(cols, ax):
    sns.boxplot(x=col, data=df, ax=i)


# In[15]:


# boxplot by class
fig, ax = plt.subplots(nrows=12, ncols=3, figsize=(25, 50))
ax = ax.ravel()

for col, i in zip(cols, ax):
    sns.boxplot(x='column_33', y=col, data=df, ax=i).set(xlabel=None)


# In[16]:


sns.pairplot(df, hue='column_33', diag_kind="hist", corner=True)
plt.show()


# In[17]:


# finding outliers with z-score 3-sd
df_no_outliers = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
non_premium_outliers = df['column_33'].value_counts()[0] - df_no_outliers['column_33'].value_counts()[0]
premium_outliers = df['column_33'].value_counts()[1] - df_no_outliers['column_33'].value_counts()[1]

print('non-premium outliers = ', non_premium_outliers)
print('premium outliers = ', premium_outliers)


# In[18]:


# count outliers by column
df_z_score = np.abs(stats.zscore(df))

for col in cols:
    count = len(df_z_score[col][df_z_score[col] > 3])
    print(col, ' = ', count)


# ## Feature Engineering

# In[3]:


# split data into train and test
y = df['column_33']
X = df.drop(['column_33'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

print(f'X_train shape:', X_train.shape)
print(f'y_train shape:', y_train.shape)
print(f'X_test shape:', X_test.shape)
print(f'y_test shape:', y_test.shape)


# In[4]:


# target encoding
cols_to_encode = ['column_27', 'column_56', 'column_73']
encoder = TargetEncoder(cols=cols_to_encode, return_df=True)
X_train_e = encoder.fit_transform(X_train, y_train)
X_test_e = encoder.transform(X_test)


# In[5]:


# oversampling methods for the encoded training dataset
smote = SMOTE(random_state=7)
X_smote, y_smote = smote.fit_resample(X_train_e, y_train)

smoteenn = SMOTEENN(random_state=7)
X_smoteenn, y_smoteenn = smoteenn.fit_resample(X_train_e, y_train)

smotetomek = SMOTETomek(random_state=7)
X_smotetomek, y_smotetomek = smotetomek.fit_resample(X_train_e, y_train)


# In[6]:


# undersampling methods for the encoded training dataset
under = RandomUnderSampler(random_state=7)
X_under, y_under = under.fit_resample(X_train_e, y_train)


# In[9]:


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
ax = ax.ravel()
ax[0].title.set_text('IMBALANCED')
ax[1].title.set_text('UNDERSAMPLED')
ax[3].title.set_text('SMOTE')
ax[4].title.set_text('SMOTEENN')
ax[5].title.set_text('SMOTETOMEK')


sns.countplot(x=y_train, ax=ax[0]).bar_label(sns.countplot(x=y_train, ax=ax[0]).containers[0])
sns.countplot(x=y_under, ax=ax[1]).bar_label(sns.countplot(x=y_under, ax=ax[1]).containers[0])
sns.countplot(x=y_smote, ax=ax[3]).bar_label(sns.countplot(x=y_smote, ax=ax[3]).containers[0])
sns.countplot(x=y_smoteenn, ax=ax[4]).bar_label(sns.countplot(x=y_smoteenn, ax=ax[4]).containers[0])
sns.countplot(x=y_smotetomek, ax=ax[5]).bar_label(sns.countplot(x=y_smoteenn, ax=ax[5]).containers[0])
plt.show()


# ## Logistic Regression Model

# ### Logistic Regression with Imbalanced, Oversampled and Undersampled Dataset

# In[7]:


def logreg(X_train=X_train, 
           y_train=y_train, 
           X_test=X_test, 
           y_test=y_test, 
           solver='lbfgs', 
           weight=None, 
           penalty='l2', 
           C=1, 
           max_iter=100, 
           title='LR', 
           manual_roc_threshold=0, 
           manual_pr_threshold=0):
    """
    Nested function. It returns all the necessary metrics for analyzing the results 
    of Logistic Regression. It starts with the Logistic Regression algorithm, 
    then produce the confusion matrix, the roc curve, the precision-recall curve, 
    it calculates the optimal thresholds using g-means and f1 score and 
    also produces the confusion matrix in those scenarios.
    """
    # logistic regression
    logreg = LogisticRegression(solver=solver, 
                                class_weight=weight, 
                                max_iter=max_iter, 
                                penalty=penalty, 
                                C=C)
    logreg.fit(X_train, y_train)

    # accuracy and log-loss
    print('Train Accuracy: ', round(logreg.score(X_train, y_train), 4))
    print('Test Accuracy: ', round(logreg.score(X_test, y_test), 4))
    print('Train log-loss: ', round(log_loss(y_train, logreg.predict_proba(X_train)), 4))
    print('Test log-loss: ', round(log_loss(y_test, logreg.predict_proba(X_test)), 4))

    # title for confusion matrix
    plot_title = title
    plot_title_1 = title + ' G-mean'
    plot_title_2 = title + ' F1'

    # predictions for the dependent variables
    y_pred = logreg.predict(X_test)
    
    # predicted probabilities for auc_score and roc curve
    y_predict_proba = logreg.predict_proba(X_test)
    
    # area under curve
    auc_score = roc_auc_score(y_test, y_predict_proba[:, 1])
    print('auc: ', round(auc_score, 4))
    
    # classification report
    print(classification_report(y_test, y_pred))

    def conf_matrix_metrics(y_test, y_pred, plot_title):
        """
        Returns the confusion matrix and its metrics sensitivity, 
        specificity and so on.
        """
        # generate confusion matrix
        confusion_m = confusion_matrix(y_test, y_pred)
        # create confusion matrix graph
        group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        group_counts = ['{0:0.0f}'.format(value) for value in confusion_m.flatten()]
        labels = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names, group_counts)]
        labels = np.asarray(labels).reshape(2, 2)
        plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_m, annot=labels, linewidths=0.01, fmt='', 
                    cmap="Blues", cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {plot_title}')
        plt.show()

        # true positives, false positives, false negatives, true negatives
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        # accuracy
        accuracy = (tp + tn) / (tp + tn + fn + fp)
        print('accuracy: ', round(accuracy, 4))
        # error rate
        error_rate = (fp + fn) / (tp + tn + fn + fp)
        print('classification error: ', round(error_rate, 4))
        # sensitivity
        sensitivity = tp / (tp + fn)
        print('sensitivity, recall or tpr: ', round(sensitivity, 4))
        # specificity
        specificity = tn / (tn + fp)
        print('specificity or tnr: ', round(specificity, 4))
        # precision
        precision = tp / (tp + fp)
        print('precision: ', round(precision, 4))
        # false positive rate (fpr)
        fpr = fp / (tn + fp)
        print('fpr (1 - specificity): ', round(fpr, 4))
        return sensitivity, precision

    # use confusion matrix function to get the metrics
    conf_matrix_results = conf_matrix_metrics(y_test, y_pred, plot_title)

    def roc_curve_plot(y_test, y_predict_proba, manual_roc_threshold=manual_roc_threshold, 
                       sensitivity=conf_matrix_results[0], auc_score=auc_score):
        """
        Returns the ROC curve with the default threshold and calculates 
        the optimal one using g-means.
        """
        # roc values
        fpr_, tpr_, thresholds_roc = roc_curve(y_test, y_predict_proba[:, 1])

        # optimal threshold roc
        gmeans = np.sqrt(tpr_*(1-fpr_))
        ix_roc = np.argmax(gmeans)

        # default threshold roc
        default_threshold_roc = np.where(np.around(tpr_, 8) == round(sensitivity, 8))
        try:
            ix_def_roc = default_threshold_roc[0][0]
        except:
            print('***Could not find your default threshold rate within the threshold array. Plot the closest one.***')
            ix_def_roc = np.abs(np.around(tpr_, 8)-round(sensitivity, 8)).argmin()

        # roc plot
        plt.figure('ROC', figsize=(7,7))
        plt.plot(fpr_, tpr_, label='Logistic (AUC = %0.3f)' % auc_score)
        plt.plot([0, 1], [0, 1], linestyle='dashed', color='red')
        # plot the optimal threshold using G-means
        plt.scatter(fpr_[ix_roc], tpr_[ix_roc], marker='o', color='black',
                    label='G-mean threshold= %0.3f' % thresholds_roc[ix_roc])
        # plot the default threshold
        plt.scatter(fpr_[ix_def_roc], tpr_[ix_def_roc], marker='o', color='red',
                    label='Default threshold= %0.3f' % thresholds_roc[ix_def_roc])
        # plot the manual threshold if provided
        if manual_roc_threshold != 0:
            ix_manual_roc = np.where(manual_roc_threshold == thresholds_roc)
            plt.scatter(fpr_[ix_manual_roc], tpr_[ix_manual_roc], marker='o', color='green',
                        label='Cost-sensitive threshold= %0.3f' % thresholds_roc[ix_manual_roc])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {plot_title}')
        plt.legend()
        #print('Optimal threshold ROC=%f, G-mean=%3f' % (thresholds_roc[ix_roc], gmeans[ix_roc]))
        
        # data for manual roc threshold
        data_roc = {'FPR': np.around(fpr_, 4)*100,
                    'TPR': np.around(tpr_, 4)*100,
                    'Thresholds ROC': thresholds_roc,
                    'FP': fpr_*3416,
                    'TP': tpr_*487,
                    'TN': 3416 - fpr_*3416,
                    'FN': 487 - tpr_*487}
        data_threshold_roc = pd.DataFrame(data_roc)
        return thresholds_roc[ix_roc]


    # use the roc curve function
    roc_curve_results = roc_curve_plot(y_test, y_predict_proba)

    # obtain the probabilities 
    y_pred_optimal_roc = (logreg.predict_proba(X_test)[:, 1] >= roc_curve_results).astype(int)

    print('')
    print('************** CONFUSION MATRIX WITH G-MEAN THRESHOLD **************')
    print('')
    # use confusion matrix function to get the metrics with the optimal threshold using g-means
    conf_matrix_metrics(y_test, y_pred_optimal_roc, plot_title_1)

    def pr_curve_plot(y_test, y_predict_proba, manual_pr_threshold=manual_pr_threshold, 
                      precision=conf_matrix_results[1]):
        """
        Returns the precision-recall curve with the default threshold and 
        calculates the optimal one using F1.
        """
        # precision-recall values
        precision_, recall_, thresholds_recall = precision_recall_curve(y_test, y_predict_proba[:, 1])

        # optimal threshold f1
        fscore = np.divide((2 * precision_ * recall_), (precision_ + recall_), out=np.zeros_like(recall_),
                           where=precision_ != 0)
        ix_f = np.argmax(fscore)
        auc_recall = auc(recall_, precision_)
        #print('Optimal threshold PR=%f, F-measure=%3f'% (thresholds_recall[ix_f], fscore[ix_f]))

        # default threshold precision-recall
        default_threshold_precision = np.where(np.around(precision_, 8) == round(precision, 8))
        try:
            ix_def_prec = default_threshold_precision[0][0]
        except:
            print('***Could not find your default threshold rate within the threshold array. Plot the closest one.***')
            ix_def_prec = np.abs(np.around(precision_, 8)-round(precision, 8)).argmin()

        # precision-recall curve
        no_skill = len(y_test[y_test == 1]) / len(y_test)
        plt.figure('Precision-Recall Curve', figsize=(7,7))
        plt.plot([0, 1], [no_skill, no_skill], linestyle='dashed', color='red')
        plt.plot(recall_, precision_, label='Logistic (AUC = %0.3f)' % auc_recall)
        # plot the optimal threshold using F1
        plt.scatter(recall_[ix_f], precision_[ix_f], marker='o', color='black',
                    label='F1 threshold= %0.3f' % thresholds_recall[ix_f])
        # plot the default threshold
        plt.scatter(recall_[ix_def_prec], precision_[ix_def_prec], marker='o', color='red',
                    label='Default threshold= %0.3f' % thresholds_recall[ix_def_prec])
        # plot the manual threshold if provided
        if manual_pr_threshold != 0:
            ix_manual_pr = np.where(manual_pr_threshold == thresholds_recall)
            plt.scatter(recall_[ix_manual_pr], precision_[ix_manual_pr], marker='o', color='green',
                        label='Cost-sensitive threshold= %0.3f' % thresholds_recall[ix_manual_pr])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve - {plot_title}')
        plt.legend()

        # data for manual precision_recall threshold
        data_pr = {'Recall': np.around(recall_, 4)*100,
                   'Precision': np.around(precision_, 4)*100,
                   'Thresholds PR': thresholds_recall,
                   'FP': (recall_*487/precision_)-recall_*487,
                   'TP': recall_*487,
                   'TN': 3416 - (recall_*487/precision_)-recall_*487,
                   'FN': 487 - recall_*487}

        data_threshold_pr = pd.DataFrame.from_dict(data_pr, orient='index').transpose()
        return thresholds_recall[ix_f]


    # optimal threshold precision recall
    pr_curve_results = pr_curve_plot(y_test, y_predict_proba)
    
    # obtain the probabilities of the optimal threshold f1
    y_pred_optimal_pr = (logreg.predict_proba(X_test)[:, 1] >= pr_curve_results).astype(int)

    print('')
    print('************** CONFUSION MATRIX WITH F1 THRESHOLD **************')
    print('')
    # use confusion matrix function to get the metrics with the optimal threshold precision recall
    conf_matrix_metrics(y_test, y_pred_optimal_pr, plot_title_2)


# In[8]:


# logistic regression with imbalanced dataset
logreg(X_train_e, y_train, X_test_e, y_test, 
       title='LR-Imbalanced', max_iter=10000)


# In[9]:


# logistic regression with imbalanced dataset (weighted)
logreg(X_train_e, y_train, X_test_e, y_test, 
       weight='balanced', title='LR-Imbalanced(Weighted)', max_iter=10000)


# In[10]:


# logistic regression with smote dataset
logreg(X_smote, y_smote, X_test_e, y_test, 
       title='LR-SMOTE', max_iter=10000)


# In[11]:


# logistic regression with smoteenn dataset
logreg(X_smoteenn, y_smoteenn, X_test_e, y_test, 
       title='LR-SMOTEENN', max_iter=10000)


# In[12]:


# logistic regression with smotetomek dataset
logreg(X_smotetomek, y_smotetomek, X_test_e, y_test, 
       title='LR-SMOTETOMEK', max_iter=10000)


# In[13]:


# logistic regression with undersampled dataset
logreg(X_under, y_under, X_test_e, y_test, 
       title='LR-UNDER', max_iter=10000)


# ## Logistic Regression - SMOTETomek

# ### Feature Selection: Correlation Based

# In[19]:


# remove one of the most correlated variables
X_smotetomek_corr = X_smotetomek.drop(['column_63', 'column_80', 
                                       'column_24', 'column_79'], axis=1)
X_test_corr = X_test_e.drop(['column_63', 'column_80', 'column_24', 
                             'column_79'], axis=1)


# In[43]:


# logistic regression with smotetomek dataset (without correlated variables)
logreg(X_smotetomek_corr, y_smotetomek, X_test_corr, y_test, 
       title='LR-SMOTETOMEK-Corr', max_iter=10000)


# ### Feature Selection: Forward Selection

# In[22]:


# import libraries
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# In[11]:


# logistic regression with arguments of smotetomek
logistic_regression = LogisticRegression(solver='lbfgs', class_weight=None, 
                                         penalty='l2', C=1, max_iter=10000)

# SFS
sfs_logreg = SFS(
    estimator=logistic_regression,
    k_features='best',
    forward=True,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1)

sfs_logreg = sfs_logreg.fit(X_smotetomek, y_smotetomek)


# In[12]:


sfs_logreg_score = sfs_logreg.k_score_
sfs_logreg_feature_idx = sfs_logreg.k_feature_idx_
sfs_logreg_feature_names = sfs_logreg.k_feature_names_


# In[13]:


print(sfs_logreg_score)


# In[14]:


print(sfs_logreg_feature_names)


# In[21]:


X_sfs_logreg = X_smotetomek[['column_16', 'column_19', 'column_22', 'column_23', 
                             'column_24', 'column_25', 'column_27', 'column_41', 
                             'column_45', 'column_48', 'column_50', 'column_52', 
                             'column_56', 'column_58', 'column_60', 'column_63', 
                             'column_64', 'column_65', 'column_68', 'column_71', 
                             'column_73', 'column_74', 'column_79', 'column_80', 
                             'column_81', 'column_94', 'column_98']]
X_test_sfs_logreg = X_test_e[['column_16', 'column_19', 'column_22', 'column_23', 
                              'column_24', 'column_25', 'column_27', 'column_41', 
                              'column_45', 'column_48', 'column_50', 'column_52', 
                              'column_56', 'column_58', 'column_60', 'column_63', 
                              'column_64', 'column_65', 'column_68', 'column_71', 
                              'column_73', 'column_74', 'column_79', 'column_80', 
                              'column_81', 'column_94', 'column_98']]


# In[19]:


logreg(X_sfs_logreg, y_smotetomek, X_test_sfs_logreg, y_test, 
       title='LR-SMOTETOMEK-SFS', max_iter=10000)


# ### Tuning Parameters

# In[30]:


# logistic regression using sfs features
logistic_regression = LogisticRegression(class_weight=None, max_iter=10000)

# define parameters for randomized grid search
params = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C': np.logspace(-4, 4, 20)}

tuned_logistic_regression = RandomizedSearchCV(estimator=logistic_regression, 
                                               param_distributions=params, 
                                               scoring='roc_auc', 
                                               cv=5, 
                                               verbose=0, 
                                               random_state=42, 
                                               n_jobs=-1, 
                                               n_iter=100)


# In[31]:


best_logistic_regression = tuned_logistic_regression.fit(X_sfs_logreg, 
                                                         y_smotetomek)


# In[32]:


best_logistic_regression.best_params_


# In[33]:


best_logistic_regression.best_score_


# In[34]:


best_logistic_regression.best_estimator_


# In[35]:


best_logistic_regression.score(X_test_sfs_logreg, y_test)


# ### Tuning Weights

# In[10]:


# logistic regression with imbalanced dataset
logreg_weights = LogisticRegression(max_iter=10000)

# range for class weights
weights = np.linspace(0.00, 0.99, 200)

# define parameters for grid search
params = {'class_weight': [{0: x, 1: 1.0-x} for x in weights]}

tuned_logreg_weights = GridSearchCV(estimator=logreg_weights, 
                                    param_grid=params, 
                                    scoring='roc_auc', 
                                    cv=StratifiedKFold(), 
                                    verbose=2, 
                                    n_jobs=-1)


# In[11]:


best_logreg_weights = tuned_logreg_weights.fit(X_train_e, 
                                               y_train)


# In[12]:


best_logreg_weights.best_params_


# In[13]:


best_logreg_weights.best_score_


# In[14]:


best_logreg_weights.best_estimator_


# In[15]:


best_logreg_weights.score(X_test_e, y_test)


# In[18]:


logreg(X_train_e, y_train, X_test_e, y_test, 
       weight={0: 0.16417085427135678, 1: 0.8358291457286432}, 
       title='LR-WEIGHTS', max_iter=10000)


# In[20]:


# remove one of the most correlated variables
X_train_corr = X_train_e.drop(['column_63', 'column_80', 'column_24', 'column_79'], axis=1)
X_test_corr = X_test_e.drop(['column_63', 'column_80', 'column_24', 'column_79'], axis=1)


# In[21]:


logreg(X_train_corr, y_train, X_test_corr, y_test, 
       weight={0: 0.16417085427135678, 1: 0.8358291457286432}, 
       title='LR-WEIGHTS', max_iter=10000)


# In[27]:


# Forwards feature selection
logreg_imb = LogisticRegression(
    solver='lbfgs', 
    class_weight={0: 0.16417085427135678, 1: 0.8358291457286432}, 
    penalty='l2', 
    C=1, 
    max_iter=10000)

# SFS
sfs_logreg_imb = SFS(
    estimator=logreg_imb,
    k_features='best',
    forward=True,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1)

sfs_logreg_imb = sfs_logreg_imb.fit(X_train_e, y_train)


# In[28]:


sfs_logreg_imb_score = sfs_logreg_imb.k_score_
sfs_logreg_imb_feature_idx = sfs_logreg_imb.k_feature_idx_
sfs_logreg_imb_feature_names = sfs_logreg_imb.k_feature_names_


# In[29]:


print(sfs_logreg_imb_score)


# In[30]:


print(sfs_logreg_imb_feature_names)


# In[10]:


X_train_sfs_logreg_imb = X_train_e[['column_16', 'column_19', 'column_23', 
                                    'column_24', 'column_27', 'column_36', 
                                    'column_41', 'column_45', 'column_48', 
                                    'column_52', 'column_56', 'column_60', 
                                    'column_65', 'column_68', 'column_71', 
                                    'column_73', 'column_74', 'column_80', 
                                    'column_81', 'column_82', 'column_94', 
                                    'column_98']]

X_test_sfs_logreg_imb = X_test_e[['column_16', 'column_19', 'column_23', 
                                  'column_24', 'column_27', 'column_36',
                                  'column_41', 'column_45', 'column_48', 
                                  'column_52', 'column_56', 'column_60', 
                                  'column_65', 'column_68', 'column_71', 
                                  'column_73', 'column_74', 'column_80', 
                                  'column_81', 'column_82', 'column_94', 
                                  'column_98']]


# In[32]:


logreg(X_train_sfs_logreg_imb, y_train, X_test_sfs_logreg_imb, y_test, 
       weight={0: 0.16417085427135678, 1: 0.8358291457286432}, 
       title='LR-WEIGHTS-SFS', max_iter=10000)


# In[11]:


# logistic regression using sfs features
logistic_regression_2 = LogisticRegression(max_iter=10000, 
                             class_weight={0: 0.16417085427135678, 1: 0.8358291457286432})

# define parameters for randomized grid search
params = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C': np.logspace(-4, 4, 20)}

tuned_logistic_regression_2 = RandomizedSearchCV(estimator=logistic_regression_2, 
                                               param_distributions=params, 
                                               scoring='roc_auc', 
                                               cv=StratifiedKFold(), 
                                               verbose=0, 
                                               random_state=42, 
                                               n_jobs=-1, 
                                               n_iter=100)


# In[12]:


best_logistic_regression_2 = tuned_logistic_regression_2.fit(X_train_sfs_logreg_imb, 
                                                         y_train)


# In[13]:


best_logistic_regression_2.best_params_


# In[14]:


best_logistic_regression_2.best_score_


# In[15]:


best_logistic_regression_2.best_estimator_


# In[16]:


best_logistic_regression_2.score(X_test_sfs_logreg_imb, y_test)


# ## XGBoost Model

# ### XGBoost with Imbalanced, Oversampled and Undersampled Dataset

# In[3]:


# libraries
from xgboost import XGBClassifier, plot_importance


# In[14]:


def xgbmodel(X_train=X_train, 
             y_train=y_train, 
             X_test=X_test, 
             y_test=y_test, 
             scoring='aucpr', 
             estimators=100, 
             weight=1, 
             title='XGB', 
             manual_roc_threshold=0, 
             manual_pr_threshold=0):
    
    """
    Nested function. It returns all the necessary metrics for analyzing the results
    of XGBoost. It starts with the XGBoost algorithm, then produce the confusion matrix, 
    the roc curve, the precision-recall curve, it calculates the optimal thresholds 
    using g-means and f1 score and also produces the confusion matrix in those scenarios.
    """
    # xgboost model
    xgbclf = XGBClassifier(eval_metric=scoring, 
                           scale_pos_weight=weight, 
                           use_label_encoder=False, 
                           n_estimators=estimators)
    xgbclf.fit(X_train, y_train)

    # accuracy and log-loss
    print('Train Accuracy: ', round(xgbclf.score(X_train, y_train), 4))
    print('Test Accuracy: ', round(xgbclf.score(X_test, y_test), 4))
    print('Train log-loss: ', round(log_loss(y_train, xgbclf.predict_proba(X_train)), 4))
    print('Test log-loss: ', round(log_loss(y_test, xgbclf.predict_proba(X_test)), 4))
    
    # plot importance 
    plot_importance(xgbclf, height=0.5, importance_type='gain')
    plt.show()

    # title for confusion matrix
    plot_title = title
    plot_title_1 = title + ' G-mean'
    plot_title_2 = title + ' F1'

    # predictions for the dependent variables
    y_pred = xgbclf.predict(X_test)
    
    # predicted probabilities for auc_score and roc curve
    y_predict_proba = xgbclf.predict_proba(X_test)
    
    # area under curve
    auc_score = roc_auc_score(y_test, y_predict_proba[:, 1])
    print('auc: ', round(auc_score, 4))
    
    # classification report
    print(classification_report(y_test, y_pred))

    def conf_matrix_metrics(y_test, y_pred, plot_title):
        """
        Returns the confusion matrix and its metrics sensitivity, 
        specificity and so on.
        """
        # generate confusion matrix
        confusion_m = confusion_matrix(y_test, y_pred)
        # create confusion matrix graph
        group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        group_counts = ['{0:0.0f}'.format(value) for value in confusion_m.flatten()]
        labels = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names, group_counts)]
        labels = np.asarray(labels).reshape(2, 2)
        plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_m, annot=labels, linewidths=0.01, fmt='', cmap="Blues", cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {plot_title}')
        plt.show()

        # true positives, false positives, false negatives, true negatives
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        # accuracy
        accuracy = (tp + tn) / (tp + tn + fn + fp)
        print('accuracy: ', round(accuracy, 4))
        # error rate
        error_rate = (fp + fn) / (tp + tn + fn + fp)
        print('classification error: ', round(error_rate, 4))
        # sensitivity
        sensitivity = tp / (tp + fn)
        print('sensitivity, recall or tpr: ', round(sensitivity, 4))
        # specificity
        specificity = tn / (tn + fp)
        print('specificity or tnr: ', round(specificity, 4))
        # precision
        precision = tp / (tp + fp)
        print('precision: ', round(precision, 4))
        # false positive rate (fpr)
        fpr = fp / (tn + fp)
        print('fpr (1 - specificity): ', round(fpr, 4))
        return sensitivity, precision


    # use confusion matrix function to get the metrics
    conf_matrix_results = conf_matrix_metrics(y_test, y_pred, plot_title)

    def roc_curve_plot(y_test, y_predict_proba, manual_roc_threshold=manual_roc_threshold, 
                       sensitivity=conf_matrix_results[0], auc_score=auc_score):
        """
        Returns the ROC curve with the default threshold and calculates 
        the optimal one using g-means.
        """
        # roc values
        fpr_, tpr_, thresholds_roc = roc_curve(y_test, y_predict_proba[:, 1])

        # optimal threshold roc
        gmeans = np.sqrt(tpr_*(1-fpr_))
        ix_roc = np.argmax(gmeans)

        # default threshold roc
        default_threshold_roc = np.where(np.around(tpr_, 8) == round(sensitivity, 8))
        try:
            ix_def_roc = default_threshold_roc[0][0]
        except:
            print('***Could not find your default threshold rate within the threshold array. Plot the closest one.***')
            ix_def_roc = np.abs(np.around(tpr_, 8)-round(sensitivity, 8)).argmin()

        # roc plot
        plt.figure('ROC', figsize=(7,7))
        plt.plot(fpr_, tpr_, label='Logistic (AUC = %0.3f)' % auc_score)
        plt.plot([0, 1], [0, 1], linestyle='dashed', color='red')
        # plot the optimal threshold using G-means
        plt.scatter(fpr_[ix_roc], tpr_[ix_roc], marker='o', color='black',
                    label='G-mean threshold= %0.3f' % thresholds_roc[ix_roc])
        # plot the default threshold
        plt.scatter(fpr_[ix_def_roc], tpr_[ix_def_roc], marker='o', color='red',
                    label='Default threshold= %0.3f' % thresholds_roc[ix_def_roc])
        # plot the manual threshold if provided
        if manual_roc_threshold != 0:
            ix_manual_roc = np.where(manual_roc_threshold == thresholds_roc)
            plt.scatter(fpr_[ix_manual_roc], tpr_[ix_manual_roc], marker='o', color='green',
                        label='Cost-sensitive threshold= %0.3f' % thresholds_roc[ix_manual_roc])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {plot_title}')
        plt.legend()
        #print('Optimal threshold ROC=%f, G-mean=%3f' % (thresholds_roc[ix_roc], gmeans[ix_roc]))
        
        # data for manual roc threshold
        data_roc = {'FPR': np.around(fpr_, 4)*100,
                    'TPR': np.around(tpr_, 4)*100,
                    'Thresholds ROC': thresholds_roc,
                    'FP': fpr_*3416,
                    'TP': tpr_*487,
                    'TN': 3416 - fpr_*3416,
                    'FN': 487 - tpr_*487}
        data_threshold_roc = pd.DataFrame(data_roc)
        return thresholds_roc[ix_roc]


    # use the roc curve function
    roc_curve_results = roc_curve_plot(y_test, y_predict_proba)

    # obtain the probabilities 
    y_pred_optimal_roc = (xgbclf.predict_proba(X_test)[:, 1] >= roc_curve_results).astype(int)

    print('')
    print('************** CONFUSION MATRIX WITH G-MEAN THRESHOLD **************')
    print('')
    # use confusion matrix function to get the metricx with the optimal threshold using g-means
    conf_matrix_metrics(y_test, y_pred_optimal_roc, plot_title_1)

    def pr_curve_plot(y_test, y_predict_proba, manual_pr_threshold=manual_pr_threshold, 
                      precision=conf_matrix_results[1]):
        """
        Returns the precision-recall curve with the default threshold and 
        calculates the optimal one using F1.
        """
        # precision-recall values
        precision_, recall_, thresholds_recall = precision_recall_curve(y_test, y_predict_proba[:, 1])

        # optimal threshold f1
        fscore = np.divide((2 * precision_ * recall_), (precision_ + recall_), out=np.zeros_like(recall_),
                           where=precision_ != 0)
        ix_f = np.argmax(fscore)
        auc_recall = auc(recall_, precision_)
        #print('Optimal threshold PR=%f, F-measure=%3f'% (thresholds_recall[ix_f], fscore[ix_f]))

        # default threshold precision-recall
        default_threshold_precision = np.where(np.around(precision_, 8) == round(precision, 8))
        try:
            ix_def_prec = default_threshold_precision[0][0]
        except:
            print('***Could not find your default threshold rate within the threshold array. Plot the closest one.***')
            ix_def_prec = np.abs(np.around(precision_, 8)-round(precision, 8)).argmin()

        # precision-recall curve
        no_skill = len(y_test[y_test == 1]) / len(y_test)
        plt.figure('Precision-Recall Curve', figsize=(7,7))
        plt.plot([0, 1], [no_skill, no_skill], linestyle='dashed', color='red')
        plt.plot(recall_, precision_, label='Logistic (AUC = %0.3f)' % auc_recall)
        # plot the optimal threshold using F1
        plt.scatter(recall_[ix_f], precision_[ix_f], marker='o', color='black',
                    label='F1 threshold= %0.3f' % thresholds_recall[ix_f])
        # plot the default threshold
        plt.scatter(recall_[ix_def_prec], precision_[ix_def_prec], marker='o', color='red',
                    label='Default threshold= %0.3f' % thresholds_recall[ix_def_prec])
        # plot the manual threshold if provided
        if manual_pr_threshold != 0:
            ix_manual_pr = np.where(manual_pr_threshold == thresholds_recall)
            plt.scatter(recall_[ix_manual_pr], precision_[ix_manual_pr], marker='o', color='green',
                        label='Cost-sensitive threshold= %0.3f' % thresholds_recall[ix_manual_pr])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve - {plot_title}')
        plt.legend()

        # data for manual precision_recall threshold
        data_pr = {'Recall': np.around(recall_, 4)*100,
                   'Precision': np.around(precision_, 4)*100,
                   'Thresholds PR': thresholds_recall,
                   'FP': (recall_*487/precision_)-recall_*487,
                   'TP': recall_*487,
                   'TN': 3416 - (recall_*487/precision_)-recall_*487,
                   'FN': 487 - recall_*487}

        data_threshold_pr = pd.DataFrame.from_dict(data_pr, orient='index').transpose()
        return thresholds_recall[ix_f]


    # use the precision recall curve function
    pr_curve_results = pr_curve_plot(y_test, y_predict_proba)

    # obtain the probabilities of the optimal threshold using F1
    y_pred_optimal_pr = (xgbclf.predict_proba(X_test)[:, 1] >= pr_curve_results).astype(int)

    print('')
    print('************** CONFUSION MATRIX WITH F1 THRESHOLD **************')
    print('')
    # use confusion matrix function to get the metricx with the optimal threshold using F1
    conf_matrix_metrics(y_test, y_pred_optimal_pr, plot_title_2)


# In[37]:


# xgboost with imbalanced dataset
xgbmodel(X_train_e, y_train, X_test_e, y_test, title='XGB-Imbalanced', 
         scoring='aucpr', estimators=100)


# In[23]:


# xgboost with imbalanced dataset (weighted)
xgbmodel(X_train_e, y_train, X_test_e, y_test, title='XGB-Imbalanced(Weighted)', 
         scoring='aucpr', estimators=100, weight=7.24)


# In[26]:


# xgboost with smote dataset
xgbmodel(X_smote, y_smote, X_test_e, y_test, title='XGB-SMOTE', 
         scoring='aucpr', estimators=100)


# In[27]:


# xgboost with smoteenn dataset
xgbmodel(X_smoteenn, y_smoteenn, X_test_e, y_test, title='XGB-SMOTEENN', 
         scoring='aucpr', estimators=100)


# In[29]:


# xgboost with smotetomek dataset
xgbmodel(X_smotetomek, y_smotetomek, X_test_e, y_test, title='XGB-SMOTETOMEK', 
         scoring='aucpr', estimators=100)


# In[30]:


# xgboost with undersampled dataset
xgbmodel(X_under, y_under, X_test_e, y_test, title='XGB-UNDER', 
         scoring='aucpr', estimators=100)


# ## XGB - SMOTE

# ### Feature Selection: Correlation Based

# In[44]:


# remove one of the most correlated variables
X_smote_corr = X_smote.drop(['column_63', 'column_80', 'column_24', 'column_79'], axis=1)
X_test_corr = X_test_e.drop(['column_63', 'column_80', 'column_24', 'column_79'], axis=1)


# In[45]:


# xgboost with smote dataset (without correlated variables)
xgbmodel(X_smote_corr, y_smote, X_test_corr, y_test, title='XGB-SMOTE-Corr', 
         scoring='aucpr', estimators=100)


# ### Feature Selection: Forward Selection

# In[48]:


# xgboost with arguments of smote
xgb_classifier = XGBClassifier(eval_metric='aucpr', 
                               n_estimators=100, 
                               use_label_encoder=False)

# SFS

sfs_xgb = SFS(xgb_classifier,
           k_features='best',
           forward=True,
           scoring='roc_auc',
           cv=5,
           n_jobs=-1)

sfs_xgb = sfs_xgb.fit(X_smote, y_smote)


# In[49]:


sfs_xgb_score = sfs_xgb.k_score_
sfs_xgb_feature_idx = sfs_xgb.k_feature_idx_
sfs_xgb_feature_names = sfs_xgb.k_feature_names_


# In[50]:


print(sfs_xgb_score)


# In[52]:


print(sfs_xgb_feature_names)


# In[57]:


X_sfs_xgb = X_smote[list(sfs_xgb_feature_names)]
X_test_sfs_xgb = X_test_e[list(sfs_xgb_feature_names)]


# In[59]:


xgbmodel(X_sfs_xgb, y_smote, X_test_sfs_xgb, y_test, title='XGB-SMOTE-SFS', 
         scoring='aucpr', estimators=100)


# ### Tuning Weights

# 

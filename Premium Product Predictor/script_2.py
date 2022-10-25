"""
Premium product predictor:
Build a model which can predict whether a product is premium or non-premium.

Disclaimer:
Due to confidentiality reasons, it is not possible to present or divulge all aspects of the dataset used in this
project.
"""


#%%
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
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve,\
    log_loss, precision_recall_curve, auc
# from sklearn.model_selection import RandomizedSearchCV


#%%
# reading data
df = pd.read_csv('/Users/faerazo/Downloads/premium-product-predictor/dataset.csv')

#%%
# imbalanced classes
sns.countplot(x=df['column_33']).bar_label(sns.countplot(x=df['column_33']).containers[0])
plt.show()


#%%
print('Descriptive Statistics')
df.describe().T

#%%
print('Skewness')
print(df.skew(axis=0))

#%%
print('Kurtosis')
df.kurtosis()

#%%
# visualize distributions of all variables
cols = df.columns
fig, ax = plt.subplots(nrows=9, ncols=4, figsize=(25, 35))
ax = ax.ravel()

for col, i in zip(cols, ax):
    sns.histplot(data=df[col], ax=i)

#%%
# visualize distributions of all variables by class
fig, ax = plt.subplots(nrows=9, ncols=4, figsize=(25, 35))
ax = ax.ravel()

for col, i in zip(cols, ax):
    sns.histplot(data=df, x=col, hue='column_33', ax=i)
#%%
# correlation matrix
correlation_matrix = df.corr(method='pearson').abs()
mask = np.triu(np.ones_like(correlation_matrix.corr(method='pearson')))
sns.set(rc={'figure.figsize': (30, 30)})
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask)


#%%
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


#%%
# top 10 correlations
sort_correlations(df)


#%%
# boxplot
cols = df.columns
fig, ax = plt.subplots(nrows=9, ncols=4, figsize=(25, 35))
ax = ax.ravel()

for col, i in zip(cols, ax):
    sns.boxplot(x=col, data=df, ax=i)

#%%
# boxplot by class
fig, ax = plt.subplots(nrows=9, ncols=4, figsize=(25, 35))
ax = ax.ravel()

for col, i in zip(cols, ax):
    sns.boxplot(x='column_33', y=col, data=df, ax=i)

#%%
sns.pairplot(df, hue='column_33', diag_kind="hist", corner=True)
plt.show()

#%%
# finding outliers with z-score 3-sd
df_no_outliers = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
non_premium_outliers = df['column_33'].value_counts()[0] - df_no_outliers['column_33'].value_counts()[0]
premium_outliers = df['column_33'].value_counts()[1] - df_no_outliers['column_33'].value_counts()[1]

print('non-premium outliers = ', non_premium_outliers)
print('premium outliers = ', premium_outliers)

#%%
# count outliers by column
df_z_score = np.abs(stats.zscore(df))

for col in cols:
    count = len(df_z_score[col][df_z_score[col] > 3])
    print(col, ' = ', count)

#%%
# split data into train and test
y = df['column_33']
X = df.drop(['column_33'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

print(f'X_train shape:', X_train.shape)
print(f'y_train shape:', y_train.shape)
print(f'X_test shape:', X_test.shape)
print(f'y_test shape:', y_test.shape)

#%%
# target encoding
cols_to_encode = ['column_27', 'column_56', 'column_73']
encoder = TargetEncoder(cols=cols_to_encode, return_df=True)
X_train_e = encoder.fit_transform(X_train, y_train)
X_test_e = encoder.transform(X_test)

#%%
# oversampling methods for the encoded training dataset
smote = SMOTE(random_state=7)
X_smote, y_smote = smote.fit_resample(X_train_e, y_train)

smoteenn = SMOTEENN(random_state=7)
X_smoteenn, y_smoteenn = smoteenn.fit_resample(X_train_e, y_train)

smotetomek = SMOTETomek(random_state=7)
X_smotetomek, y_smotetomek = smotetomek.fit_resample(X_train_e, y_train)

#%%
# undersampling methods for the encoded training dataset
under = RandomUnderSampler(random_state=7)
X_under, y_under = under.fit_resample(X_train_e, y_train)

#%%
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))
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


#%%
def logreg(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, solver='lbfgs',
           weight=None, penalty='l2', C=1, max_iter=100, plot_title='LR',
           manual_roc_threshold=0, manual_pr_threshold=0):
    """
    Nested function. It returns all the necessary metrics for analyzing the results of Logistic Regression. It starts
    with the Logistic Regression algorithm, then produce the confusion matrix, the roc curve, the precision-recall
    curve, it calculates the optimal thresholds using g-means and f1 score and also produces the confusion matrix in
    those scenarios.
    """
    # logistic regression
    logreg = LogisticRegression(solver=solver, class_weight=weight, max_iter=max_iter, penalty=penalty, C=C)
    logreg.fit(X_train, y_train)

    print('Train Accuracy: ', logreg.score(X_train, y_train))
    print('Test Accuracy: ', logreg.score(X_test, y_test))
    print('Train log-loss: ', log_loss(y_train, logreg.predict_proba(X_train)))
    print('Test log-loss: ', log_loss(y_test, logreg.predict_proba(X_test)))

    # predictions for the dependent variables
    y_pred = logreg.predict(X_test)
    # predicted probabilities for auc_score and roc curve
    y_predict_proba = logreg.predict_proba(X_test)
    # area under curve
    auc_score = roc_auc_score(y_test, y_predict_proba[:, 1])
    print('auc: ', round(auc_score, 4))
    # classification report
    classification_report(y_test, y_pred)

    def conf_matrix_metrics(y_test, y_pred, plot_title=plot_title):
        """
        Returns the confusion matrix and its metrics sensitivity, specificity and so on.
        """
        confusion_m = confusion_matrix(y_test, y_pred)
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


    conf_matrix_results = conf_matrix_metrics(y_test, y_pred, y_predict_proba)

    def roc_curve_plot(y_test, y_predict_proba, manual_roc_threshold=manual_roc_threshold,
                           plot_title=plot_title, sensitivity=conf_matrix_results[0], auc_score=auc_score):
        """
        Returns the ROC curve with the default threshold and calculates the optimal one using g-means.
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
        plt.figure('ROC')
        plt.plot(fpr_, tpr_, label='Logistic (AUC = %0.3f)' % auc_score)
        plt.plot([0, 1], [0, 1], linestyle='dashed', color='red')
        plt.scatter(fpr_[ix_roc], tpr_[ix_roc], marker='o', color='black',
                    label='G-mean threshold= %0.3f' % thresholds_roc[ix_roc])
        plt.scatter(fpr_[ix_def_roc], tpr_[ix_def_roc], marker='o', color='red',
                    label='Default threshold= %0.3f' % thresholds_roc[ix_def_roc])
        if manual_roc_threshold != 0:
            ix_manual_roc = np.where(manual_roc_threshold == thresholds_roc)
            plt.scatter(fpr_[ix_manual_roc], tpr_[ix_manual_roc], marker='o', color='green',
                        label='Cost-sensitive threshold= %0.3f' % thresholds_roc[ix_manual_roc])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {plot_title}')
        plt.legend()
        print('Optimal threshold ROC=%f, G-mean=%3f' % (thresholds_roc[ix_roc], gmeans[ix_roc]))
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


    # optimal threshold roc
    roc_curve_results = roc_curve_plot(y_test, y_predict_proba)

    y_pred_optimal_roc = (logreg.predict_proba(X_test_e)[:, 1] >= roc_curve_results).astype(int)

    conf_matrix_metrics(y_test, y_pred_optimal_roc, 0)

    def pr_curve_plot(y_test, y_predict_proba, manual_pr_threshold=manual_pr_threshold,
                      plot_title=plot_title, precision=conf_matrix_results[1]):
        """
        Returns the precision-recall curve with the default threshold and calculates the optimal one using F1.
        """
        # precision-recall values
        precision_, recall_, thresholds_recall = precision_recall_curve(y_test, y_predict_proba[:, 1])

        # optimal threshold precision-recall
        fscore = np.divide((2 * precision_ * recall_), (precision_ + recall_), out=np.zeros_like(recall_),
                           where=precision_ != 0)
        ix_f = np.argmax(fscore)
        auc_recall = auc(recall_, precision_)
        print('Optimal threshold PR=%f, F-measure=%3f'% (thresholds_recall[ix_f], fscore[ix_f]))

        # default threshold precision-recall
        default_threshold_precision = np.where(np.around(precision_, 8) == round(precision, 8))
        try:
            ix_def_prec = default_threshold_precision[0][0]
        except:
            print('***Could not find your default threshold rate within the threshold array. Plot the closest one.***')
            ix_def_prec = np.abs(np.around(precision_, 8)-round(precision, 8)).argmin()

        # precision-recall curve
        no_skill = len(y_test[y_test == 1]) / len(y_test)
        plt.figure('Precision-Recall Curve')
        plt.plot([0, 1], [no_skill, no_skill], linestyle='dashed', color='red')
        plt.plot(recall_, precision_, label='Logistic (AUC = %0.3f)' % auc_recall)
        plt.scatter(recall_[ix_f], precision_[ix_f], marker='o', color='black',
                    label='F1 threshold= %0.3f' % thresholds_recall[ix_f])
        plt.scatter(recall_[ix_def_prec], precision_[ix_def_prec], marker='o', color='red',
                    label='Default threshold= %0.3f' % thresholds_recall[ix_def_prec])
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

    y_pred_optimal_pr = (logreg.predict_proba(X_test)[:, 1] >= pr_curve_results).astype(int)

    conf_matrix_metrics(y_test, y_pred_optimal_pr, 0)


#%%
# logistic regression with imbalanced dataset
logreg(X_train_e, y_train, X_test_e, y_test, plot_title='LR - Imbalanced', max_iter=10000)

#%%
# logistic regression with imbalanced dataset (weighted)
logreg(X_train_e, y_train, X_test_e, y_test, weight='balanced', plot_title='LR - Imbalanced(Weighted)',
       max_iter=10000)

#%%
# logistic regression with smote dataset
logreg(X_smote, y_smote, X_test_e, y_test, plot_title='LR-SMOTE',
       max_iter=10000)

#%%
# logistic regression with smoteenn dataset
logreg(X_smoteenn, y_smoteenn, X_test_e, y_test, plot_title='LR - SMOTEENN',
       max_iter=10000)

#%%
# logistic regression with smotetomek datastet
logreg(X_smotetomek, y_smotetomek, X_test_e, y_test, plot_title='LR - SMOTETOMEK',
       max_iter=10000)

#%%
# logistic regression with undersampled datastet
logreg(X_under, y_under, X_test_e, y_test, plot_title='LR - UNDER',
       max_iter=10000)

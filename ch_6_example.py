import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def data_deal():
    df = pd.read_csv('wdbc.data', header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 1)

    '''combine transformers and estimators in a pipeline'''
    pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components = 2)), ('clf', LogisticRegression(random_state = 1))])
    pipe_lr.fit(X_train, y_train)
    # print ('Test Accuracy: {a:.3f}'.format(a = pipe_lr.score(X_test, y_test)))


    '''using k-fold cross-validation to assess model performance'''
    # holdout cross-validation 

    # k-fold cross-validation

    # stratified k-fold cv
    kfold = StratifiedKFold(y = y_train, n_folds = 10, random_state = 1)
    
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        # print ('Fold: {n}, Class dist.: {qty}, Acc: {score:.3f}'.format(n = k+1, qty = np.bincount(y_train[train]), score = score))
    
    scores = cross_val_score(estimator = pipe_lr, X = X_train, y = y_train, cv = 10, n_jobs = 1)
    # print ('CV accruracy scores: {scores}'.format(scores = scores))
    # print ('CV accuracy: {score_mean:.3f} +/- {score_std:.3f}'.format(score_mean = np.mean(scores), score_std = np.std(scores)))

    '''diagnosing bias and variace with learning curves'''
    pipe_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(penalty = 'l2', random_state = 0))])
    train_sizes, train_scores, test_scores = learning_curve(estimator = pipe_lr, X = X_train, y = y_train, train_sizes = np.linspace(.1, 1.0, 10), cv = 10, n_jobs = 1)
    
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)
    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)
    # plt.plot(train_sizes, train_mean, color = 'blue', marker = 'o', markersize = 5, label = 'training accuracy')
    # plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = .15, color = 'blue')
    # plt.plot(train_sizes, test_mean, color = 'green', linestyle = '--', marker = 's', markersize = 5, label = 'validation accuracy')
    # plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = .15, color = 'green')
    # plt.grid()
    # plt.xlabel('Number of training samples')
    # plt.ylabel('Accuracy')
    # plt.legend(loc = 'lower right')
    # plt.ylim([.8, 1.0])
    # plt.show()

    '''address overfitting/ underfitting w/ validation curves'''
    param_range = [.001, .01, .1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(estimator = pipe_lr, X = X_train, y = y_train, param_name = 'clf__C', param_range = param_range, cv = 10)
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)
    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)
    
    # plt.plot(param_range, train_mean, color = 'blue', marker = 'o', markersize = 5, label = 'training accuracy')
    # plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha = .15, color = 'blue')
    # plt.plot(param_range, test_mean, color = 'green', linestyle = '--', marker = 's', markersize = 5, label = 'validation accuracy')
    # plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha = .15, color = 'green')
    # plt.grid()
    # plt.xscale('log')
    # plt.legend(loc = 'lower right')
    # plt.xlabel('Prameter C')
    # plt.ylabel('Accuracy')
    # plt.ylim([.8, 1.0])
    # plt.show()

    '''tuning machine learning models via grid search'''
    pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state  = 1))])
    param_range = [ .0001, .001, .01, .1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'clf__C': param_range, 'clf__kernel':['linear']}, {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel':['rbf']}]
    gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)
    gs = gs.fit(X_train, y_train)
    # print(gs.best_score_)
    # print (gs.best_params_)

    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    # print ('Test accuracy: {score:.3f}'.format(score = clf.score(X_test, y_test)))

    '''algorithm selection w/ nested cv'''
    gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid, scoring = 'accuracy', cv = 2, n_jobs = -1)
    scores = cross_val_score(gs, X_train, y_train, scoring = 'accuracy', cv = 5)
    # print ('CV accuracy: {s_mean:.3f} +/- {s_std:.3f}'.format(s_mean = np.mean(scores), s_std = np.std(scores)))

    
    gs = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 0), param_grid = [{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], scoring = 'accuracy', cv = 5)
    scores = cross_val_score(gs, X_train, y_train, scoring = 'accuracy', cv = 2)
    # print ('CV accuracy: {s_mean:.3f} +/- {s_std:.3f}'.format(s_mean = np.mean(scores), s_std = np.std(scores)))

    '''different performance evaluation metrics'''
    ''''confusion matrix'''
    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true = y_test, y_pred = y_pred)
    # print (confmat)

    # fig, ax = plt.subplots(figsize = (2.5, 2.5))
    # ax.matshow(confmat, cmap = plt.cm.Blues, alpha = .3)
    # for i in range(confmat.shape[0]): 
    #     for j in range(confmat.shape[1]):
    #         ax.text(x = j, y = i, s = confmat[i, j], va = 'center', ha = 'center')
    # plt.xlabel('predicted label')
    # plt.ylabel('true label')
    # plt.show()

    '''optimize the precision/ recall of a classification model'''
    # print ('Precision: {pre:.3f}'.format(pre = precision_score(y_true = y_test, y_pred = y_pred)))

    # print ('Recall: {recall:.3f}'.format(recall = recall_score(y_true = y_test, y_pred = y_pred)))

    # print ('F1: {f1:.3f}'.format(f1 = f1_score(y_true = y_test, y_pred = y_pred)))

    scorer = make_scorer(f1_score, pos_label = 0)

    gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid, scoring = scorer, cv = 10)

    '''plot a ROC (receiver operating characteristic)'''
    pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components = 2)), ('clf', LogisticRegression(penalty = 'l2', random_state = 0, C = 100.0))])
    X_train2 = X_train[:, [4, 14]]
    cv = StratifiedKFold(y_train, n_folds = 3, random_state = 1)
    fig = plt.figure(figsize = (7, 5))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test]) 
        fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label = 1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw = 1, label = 'ROC fold {n} (area = {auc:.2f})'.format(n = i + 1, auc = roc_auc))

    plt.plot([0, 1], [0, 1], linestyle = '--', color = (.6, .6, .6), label = 'random guessing')
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label = 'mean ROC (area = {auc_area:.2f})'.format(auc_area = mean_auc), lw = 2)
    plt.plot([0, 0, 1], [0, 1, 1], lw = 2, linestyle = ':', color = 'black', label = 'perfect performance')
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic')
    plt.legend(loc = 'lower right')
    plt.show()


if __name__ == '__main__':
    data_deal()
    # stopped at page 187/ 212
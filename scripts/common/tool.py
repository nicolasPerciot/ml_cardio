import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    
    

## ---------------------------Dataset Split Train Test--------------------------- ##

## Return index to split the dataset in train and test
def dataset_split_index(X, y, test_size = 0.2, fold=None, random_state=42) :
    
    index_train_list    = []
    index_test_list     = []
    
    if fold == None :
        data_train, data_test, labels_train, labels_test, index_train, index_test = train_test_split(X, y, X.index, test_size=test_size, random_state=random_state)
        index_train_list.append(index_train)
        index_test_list.append(index_test)
    else :
        kf          = KFold(n_splits=fold, random_state=random_state, shuffle=True)
        for i, (index_train, index_test) in enumerate(kf.split(X)) :
            index_train_list.append(index_train)
            index_test_list.append(index_test)
    return index_train_list, index_test_list


## Return list train and test (features (X) and target (y))
def split_train(X, y, index_train, index_test) :
    
    if isinstance(X, pd.DataFrame) :
        X_train = X.iloc[index_train, :]
        y_train = y[index_train]
        X_test  = X.iloc[index_test, :]
        y_test  = y[index_test]
    else :
        X_train = X[index_train]
        y_train = y[index_train]
        X_test  = X[index_test]
        y_test  = y[index_test]
    return X_train, y_train, X_test, y_test



## ---------------------------Preprocessing--------------------------- ##

## Return list names of numerical columns with binary and categorical columns in input
def find_numeric_columns(data, binary_columns=[], categorical_columns=[]) :
    numerical_columns = []
    for elem in data.columns.to_list():
        if elem not in binary_columns and elem not in categorical_columns:
            numerical_columns.append(elem)
    return numerical_columns


## Find the number of element to do a PCA
def find_nb_pca(data, numerical_columns) :
    data_sample = data.loc[:, numerical_columns]
    data_sum = data_sample.sum().sort_values(axis=0, ascending = False)
    sum = 0
    for value in data_sum :
        sum += value
        
    value_perc = []

    sum_perc = 0
    for i in range(len(data_sum)) :
        
        perc = data_sum[i] / sum
        value_perc.append(perc)
        sum_perc += perc
        if sum_perc >= 0.95 :
            break
    return i + 1
            
      
## Make the right preprocessor
def make_preprocess(X, binary_columns=[], categorical_columns=[]) :

    numerical_columns = find_numeric_columns(X, binary_columns, categorical_columns)
    
    numerical_transformer       = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=find_nb_pca(X, numerical_columns))),    
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('encoder', OneHotEncoder())
    ])

    if categorical_columns != [] :
        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numerical_transformer, numerical_columns),
                ('categorical', categorical_transformer, categorical_columns),
            ])
    else :
        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numerical_transformer, numerical_columns),
            ])
    return preprocessor



## Return the preprocessor and the features transform
def preprocessing(X, binary_columns=[], categorical_columns=[]) :
    preprocessor = make_preprocess(X, binary_columns, categorical_columns)
    
    X = preprocessor.fit_transform(X)
    
    return preprocessor, X



## ---------------------------Find the best grid--------------------------- ##

## Find the best features for all the folds return a dict with the metrics
def find_best_grid_fold(model, X, y, grid_param, index_train, index_test, k_neighbors_smote=None, scoring='accuracy') :
    
    score_list      = []
    
    dict_scoring    = {
        'accuracy'      : accuracy_score,
        'precision'     : precision_score,
        'recall'        : recall_score,
        'f1'            : f1_score
    }
    
    report          = {
        'mean_score'        : 0,
        'st_score'          : 0,
        'best_score'        : 0,
        'best_grid'         : grid_param,
        'best_confusion'    : "",
        'fold'              : {}
    }
    
    for i in range(len(index_train)) :
                              
        X_train, y_train, X_test, y_test = split_train(X, y, index_train[i], index_test[i])
        
        if k_neighbors_smote != None :
            sm = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
            X_train, y_train = sm.fit_resample(X_train, y_train)
    
        
    
        model.set_params(**grid_param)
        model.fit(X_train, y_train)
        
        y_pred  = model.predict(X_test)
        score   = dict_scoring[scoring](y_test, y_pred)
        
        score_list.append(score)
        
        report['fold']['fold{}'.format(i + 1)] = {
            'score'        : score,
            'confusion'    : confusion_matrix(y_test, y_pred),
        }
        
        if score > report['best_score'] :
            report['best_score']        = score
            report['best_confusion']    = confusion_matrix(y_test, y_pred)
            index_fold                  = i
            
    report['mean_score']    = np.mean(score_list)
    report['st_score']      = np.std(score_list)
            
    return report, index_fold



## Find the best features for all the different parameters return a dict with the metrics
def find_best_grid(model, X, y, index_train, index_test, k_neighbors_smote=None, param=None, scoring='accuracy') :
    
    report          = {
        'best_mean_score'           : 0,
        'best_st_score'             : 0,
        'best_score'                : 0,
        'best_grid'                 : "",
        'best_confusion'            : "",
        'report'                    : []
    }
       
    
    if param == None :        
        report_fold = find_best_grid_fold(model, X, y, model.get_params(), index_train=index_train, 
                                 index_test=index_test, k_neighbors_smote=k_neighbors_smote, scoring=scoring)
        
        report['best_mean_score']   = report_fold['mean_score']
        report['best_st_score']     = report_fold['st_score']
        report['best_score']        = report_fold['best_score']
        report['best_grid']         = report_fold['best_grid']
        report['best_confusion']    = report_fold['best_confusion']
        
        report['report'].append(report_fold)
        
    else :
        for g in ParameterGrid(param):
            try :
                report_fold, index_fold = find_best_grid_fold(
                model, X, y, g, index_train=index_train, index_test=index_test, 
                k_neighbors_smote=k_neighbors_smote, scoring=scoring)
            except :
                continue
                
            report['report'].append(report_fold)
            if report_fold['mean_score'] > report['best_mean_score'] :
                report['best_mean_score']   = report_fold['mean_score']
                report['best_st_score']     = report_fold['st_score']
                report['best_score']        = report_fold['best_score']
                report['best_grid']         = report_fold['best_grid']
                report['best_confusion']    = report_fold['best_confusion']
                best_index_fold             = index_fold
            
    return report, best_index_fold
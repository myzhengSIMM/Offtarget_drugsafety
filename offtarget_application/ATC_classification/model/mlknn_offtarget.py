
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from skmultilearn.adapt import MLkNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
import os
import warnings
warnings.filterwarnings("ignore")

from utils_metric import get_metrics

'''
mlknn's python project
@Ml-knn: A Lazy Learning Approach to Multi-Label Learning by Min-Ling Zhang, Zhi-Hua Zhou
'''
'''
get train and test data
from yeast
you can download data set from http://archive.ics.uci.edu/ml/datasets/Yeast
'''
def search_predict_result(smiles,test_predict_result_path):
    index_list = []
    test_predict_result_df = pd.read_csv(test_predict_result_path)
    for i in range(len(test_predict_result_df)):
        for smile in smiles:
            if test_predict_result_df.at[i,'smiles'] == smile:
                index_list.append(i)
    test_predict_result_df = test_predict_result_df.iloc[index_list]
    test_predict = test_predict_result_df.values[:,1:] 
    test_predict_result = np.matrix(test_predict) 
    
    return test_predict_result


def get_train_test():
    print("start to load data...")
    
    data = pd.read_csv('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data/atc_drug_smiles_label.csv') 
    data_train = pd.read_csv('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data/train.csv') 
    data_test = pd.read_csv('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data/test.csv')
    atc_labels = data.columns[1:]
    data_train_smiles = data_train.smiles
    data_train_y = data_train.iloc[:,1:] 
    data_train_y = np.matrix(data_train_y) 
    
    data_test_smiles = data_test.smiles
    data_test_y = data_test.iloc[:,1:] 
    data_test_y = np.matrix(data_test_y) 

    test_predict_result_path = '/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data/ATC_prediction.csv' #ATC code prediction result
    data_train_x = search_predict_result(data_train_smiles,test_predict_result_path)
    data_test_x = search_predict_result(data_test_smiles,test_predict_result_path)
    
    print("load data finished...")
    
    print("start to split data...")
    data_train_x = data_train_x.A
    data_train_y = data_train_y.A
    data_test_x = data_test_x.A
    data_test_y = data_test_y.A

    if os.path.exists('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data_offtarget/X_train.npy'):
        pass
    else:
        np.save('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data_offtarget/X_train.npy', data_train_x, allow_pickle=True) 
        np.save('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data_offtarget/y_train.npy', data_train_y, allow_pickle=True)
        np.save('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data_offtarget/X_test.npy', data_test_x, allow_pickle=True)
        np.save('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data_offtarget/y_test.npy', data_test_y, allow_pickle=True)
    
    return atc_labels 



    
if __name__ == "__main__":
    
    atc_labels = get_train_test() 
    X_train = np.load('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data_offtarget/X_train.npy', allow_pickle=True)
    y_train = np.load('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data_offtarget/y_train.npy', allow_pickle=True)
    X_test = np.load('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data_offtarget/X_test.npy', allow_pickle=True)
    y_test = np.load('/home/liujin/Offtarget_drugsafety/offtarget_application/ATC_classification/data_offtarget/y_test.npy', allow_pickle=True)

    rank_loss = []
    avg_pre = []
    roc_auc = []
    iters = 0
    
    print("start to 5 kfold validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=2018) 

    for train_index, val_index in kf.split(X_train): 
        iters += 1

        # split dataset into train and validation dataset
        train_X = X_train[train_index]
        val_X = X_train[val_index]
        train_y = y_train[train_index]
        val_y = y_train[val_index]

        classifier = MLkNN(k=3,s=0.7,ignore_first_neighbours=1)

        print("start to %d training..." % (iters))
        classifier.fit(train_X, train_y)
        print("%d training finished..." % (iters))

        print("start to %d validation..." % (iters))
        val_predict_labels = classifier.predict(val_X) 
        val_predict_probs = classifier.predict_proba(val_X)
        val_predict_labels = val_predict_labels.toarray() 
        val_predict_probs = val_predict_probs.toarray()
        print("%d validation finished..." % (iters))
        

        print("start record metric value...")
       
        predict_labels = classifier.predict(X_test) # predict test dataset
        predict_probs = classifier.predict_proba(X_test) 
        predict_labels = predict_labels.toarray() 
        predict_probs = predict_probs.toarray() 

        _, _, rl, avg_precision = get_metrics(y_test, predict_labels, predict_probs) 
        rank_loss.append(rl)
        avg_pre.append(avg_precision)

        try:
            test_roc_auc = roc_auc_score(y_test,predict_probs,
                                            average='micro')
        except Exception as e:
            print("Error computing training ROC AUC:",e)
        roc_auc.append(test_roc_auc)

        print('end record...')
    
    
    print("test_rank loss: mean is %f, std is %f." % (np.mean(np.array(rank_loss)), np.std(np.array(rank_loss))))
    print("test_average precision: mean is %f, std is %f." % (np.mean(np.array(avg_pre)), np.std(np.array(avg_pre))))
    print("test_roc_auc: mean is %f, std is %f." % (np.mean(np.array(roc_auc)), np.std(np.array(roc_auc))))
    print(roc_auc)
    print(avg_pre)
    print(rank_loss)


    try:
        f = open("results_mlknn_offtarget.txt", encoding="utf8", mode='w')  
        f.write("test_rank loss: mean is %f, std is %f.\n" % (np.mean(np.array(rank_loss)), np.std(np.array(rank_loss))))
        f.write("test_average precision: mean is %f, std is %f.\n" % (np.mean(np.array(avg_pre)), np.std(np.array(avg_pre))))
        f.write("test_roc_auc: mean is %f, std is %f.\n" % (np.mean(np.array(roc_auc)), np.std(np.array(roc_auc))))
    except:
        print("file write error!")
    finally:
        f.close()
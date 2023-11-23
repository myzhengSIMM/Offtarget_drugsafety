import os
import sys

sys.path.append('/home/liujin/Offtarget_drugsafety/drugsafety')
from munch import DefaultMunch
import torch
from argparse import ArgumentParser
from model.attentive_fp import AttentiveFPModel

from dataset_builder.build_dcdataset import dcdata_loader, dcdata_withdraw_loader,dcdata_toxic_loader
import pandas as pd
import numpy as np


import warnings
warnings.filterwarnings("ignore")
import linecache
import pathlib


def run_predicting(csv_file_path):
    
    
    _,input_dataset = dcdata_withdraw_loader(csv_file_path, featurizer='GraphConv', reload=True) 

    
    header = linecache.getline(dataset_chembl_file, 1).strip('\n').split(',')  
    tasks = header[1:]  # get the target names under each model

    smiles_list = []
    for i in input_dataset.ids:
        smiles_list.append(i)

    tasks_list = []
    for j in tasks:
        tasks_list.append(j)
    
    model = AttentiveFPModel(
        device=config.device,
        mode='classification',
        num_layers=config.num_layer, 
        num_timesteps=config.num_timesteps,
        graph_feat_size=config.graph_feat_size,
        n_tasks=len(tasks),
        batch_size=config.batch_size,
        learning_rate=config.lr,
        dropout=config.dropout,
    )
    

    print(f'now is {model_project} and fold is 0')
    model_path_0 = os.path.join(result_dir, model_project, f'cross_validation_0', 'best_model')
    model.restore(model_dir = model_path_0)
    test_dataset_predict_0 = prediction_drug(model,input_dataset)

    print(f'now is {model_project} and fold is 1')
    model_path_1 = os.path.join(result_dir, model_project, f'cross_validation_1', 'best_model')
    model.restore(model_dir = model_path_1)
    test_dataset_predict_1 = prediction_drug(model,input_dataset)

    print(f'now is {model_project} and fold is 2')
    model_path_2 = os.path.join(result_dir, model_project, f'cross_validation_2', 'best_model')
    model.restore(model_dir = model_path_2)
    test_dataset_predict_2 = prediction_drug(model,input_dataset)

    print(f'now is {model_project} and fold is 3')
    model_path_3 = os.path.join(result_dir, model_project, f'cross_validation_3', 'best_model')
    model.restore(model_dir = model_path_3)
    test_dataset_predict_3 = prediction_drug(model,input_dataset)

    print(f'now is {model_project} and fold is 4')
    model_path_4 = os.path.join(result_dir, model_project, f'cross_validation_4', 'best_model')
    model.restore(model_dir = model_path_4)
    test_dataset_predict_4 = prediction_drug(model,input_dataset)

    test_dataset_predict = test_dataset_predict_0 + test_dataset_predict_1 + test_dataset_predict_2 + test_dataset_predict_3 + test_dataset_predict_4

    test_dataset_predict = test_dataset_predict / 5 
    
    return test_dataset_predict,smiles_list,tasks_list

def prediction_drug(model,dataset):
    test_prediction = model.predict(dataset)[:, :, 1]
    return test_prediction


if __name__ == "__main__":

    dataset_dir ='/home/liujin/Offtarget_drugsafety/databases/model_dataset' # The save path of model training data
    predict_drug_dir = '/home/liujin/Offtarget_drugsafety/databases/external_predict_data' # The path where the file to be predicted
    result_dir = f'/home/liujin/Offtarget_drugsafety/drugsafety/results/data_combine_train' #The path where the model is saved

    csv_file_path = os.path.join(predict_drug_dir, 'atc_drug_smiles_label.csv') # Get the file to predict,user can change the file name
    results_save_dir = '/home/liujin/Offtarget_drugsafety/drugsafety/predict/predict_result' # The path where the prediction results are saved
    save_path = os.path.join(results_save_dir,'predict_ATC_result') # User can change the folder name of the prediction results
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 
    
    
    for model_project in [
                          'parsedGPCR_3decoys_1663475547_random_stratified',
                          'parsedKinases_3decoys_1663421817_random_stratified',
                          'parsedNR_3decoys_1663704378_random_stratified',
                          ]:
        
        config = DefaultMunch.fromDict({
        "uncertainty":"None",
        "split": 'random_stratified',
        "device": "cuda:0",
        "dropout": 0.1, 
        "num_layer": 3, 
        "num_timesteps": 2, 
        "graph_feat_size": 200, 
        "lr": 0.0001,
        "batch_size": 128,
        "epochs": 100
        })

        dataset_name = f'{model_project.split("_")[0]}_3decoys.csv'
        target_class = model_project.split('_')[0].split('ed')[-1] # Get the model target class
        dataset_chembl_file = os.path.join(dataset_dir, dataset_name)

        dataset_predict,smiles_list,tasks_list = run_predicting(csv_file_path)
        prediction_df = pd.DataFrame(dataset_predict,index=smiles_list,columns=tasks_list)
        prediction_df.to_csv(os.path.join(save_path,f'{target_class}_prediction.csv'))
    
    for model_project in ['parsedEnzyme_3decoys_1676031919_random_stratified',]:

        config = DefaultMunch.fromDict({
        "uncertainty":"None",
        "split": 'random_stratified',
        "device": "cuda:0",
        "dropout": 0.1, 
        "num_layer": 1, 
        "num_timesteps": 4, 
        "graph_feat_size": 600, 
        "lr": 0.0001,
        "batch_size": 128,
        "epochs": 100
        })

        dataset_name = f'{model_project.split("_")[0]}_3decoys.csv'
        target_class = model_project.split('_')[0].split('ed')[-1] 
        dataset_chembl_file = os.path.join(dataset_dir, dataset_name)

        dataset_predict,smiles_list,tasks_list = run_predicting(csv_file_path)
        prediction_df = pd.DataFrame(dataset_predict,index=smiles_list,columns=tasks_list)
        prediction_df.to_csv(os.path.join(save_path,f'{target_class}_prediction.csv'))
    
    for model_project in ['parsedIonchannel_5decoys_1663648519_random_stratified',
                          'parsedTransporter_5decoys_1663443625_random_stratified',
                          'parsedother_5decoys_1663742473_random_stratified']:
        
        config = DefaultMunch.fromDict({
        "uncertainty":"None",
        "split": 'random_stratified',
        "device": "cuda:0",
        "dropout": 0.1, 
        "num_layer": 3, 
        "num_timesteps": 2, 
        "graph_feat_size": 200, 
        "lr": 0.0001,
        "batch_size": 128,
        "epochs": 100
        })
        
        dataset_name = f'{model_project.split("_")[0]}_5decoys.csv'
        target_class = model_project.split('_')[0].split('ed')[-1] 
        dataset_chembl_file = os.path.join(dataset_dir, dataset_name)

        dataset_predict,smiles_list,tasks_list = run_predicting(csv_file_path)
        prediction_df = pd.DataFrame(dataset_predict,index=smiles_list,columns=tasks_list)
        prediction_df.to_csv(os.path.join(save_path,f'{target_class}_prediction.csv'))


    # Merge the prediction results of all targets
    target_name = ['GPCR','Ionchannel','Enzyme','Kinases','NR','Transporter','other']

    predict_result_file_d = []
    for j in target_name:
        predict_result_file_d.append(os.path.join(save_path,f'{j}_prediction.csv'))
    
    predict_result_df0 = pd.read_csv(predict_result_file_d[0])
    predict_result_df1 = pd.read_csv(predict_result_file_d[1])
    predict_result_df2 = pd.read_csv(predict_result_file_d[2])
    predict_result_df3 = pd.read_csv(predict_result_file_d[3])
    predict_result_df4 = pd.read_csv(predict_result_file_d[4])
    predict_result_df5 = pd.read_csv(predict_result_file_d[5])
    predict_result_df6 = pd.read_csv(predict_result_file_d[6])
    predict_result_df0.rename(columns={'Unnamed: 0':'smiles'}, inplace = True)
    predict_result_df1.rename(columns={'Unnamed: 0':'smiles'}, inplace = True)
    predict_result_df2.rename(columns={'Unnamed: 0':'smiles'}, inplace = True)
    predict_result_df3.rename(columns={'Unnamed: 0':'smiles'}, inplace = True)
    predict_result_df4.rename(columns={'Unnamed: 0':'smiles'}, inplace = True)
    predict_result_df5.rename(columns={'Unnamed: 0':'smiles'}, inplace = True)
    predict_result_df6.rename(columns={'Unnamed: 0':'smiles'}, inplace = True)
    predict_result_df_all = pd.merge(predict_result_df0,predict_result_df1,on='smiles',how='outer')
    predict_result_df_all = pd.merge(predict_result_df_all,predict_result_df2,on='smiles',how='outer')
    predict_result_df_all = pd.merge(predict_result_df_all,predict_result_df3,on='smiles',how='outer')
    predict_result_df_all = pd.merge(predict_result_df_all,predict_result_df4,on='smiles',how='outer')
    predict_result_df_all = pd.merge(predict_result_df_all,predict_result_df5,on='smiles',how='outer')
    predict_result_df_all = pd.merge(predict_result_df_all,predict_result_df6,on='smiles',how='outer')
    print("Predict result shape after merging: ",predict_result_df_all.shape)
    predict_result_df_all.to_csv(os.path.join(save_path, f'alltarget_prediction.csv'),index=False) # The filename the prediction results are saved,user can change it
import os
import deepchem
import linecache
from deepchem.data import DiskDataset

# Define the featurizers to be used in the dataset processing
featurizers = {
    'ECFP': deepchem.feat.CircularFingerprint(size=1024),
    'GraphConv': deepchem.feat.MolGraphConvFeaturizer(use_edges=True),  # Suitable for Graph Neural Networks
    'Weave': deepchem.feat.WeaveFeaturizer(),
    'Raw': deepchem.feat.RawFeaturizer(),
}

# Define various dataset splitting methods
splitters = {
    'index': deepchem.splits.IndexSplitter(),
    'random': deepchem.splits.RandomSplitter(),
    'random_stratified': deepchem.splits.RandomStratifiedSplitter(),
    'scaffold': deepchem.splits.ScaffoldSplitter(),
    'butina': deepchem.splits.ButinaSplitter(),
    'task': deepchem.splits.TaskSplitter()
}


# Function to load and process dataset using DeepChem
def dcdata_loader(dataset_file, featurizer='GraphConv', transformers=True,
                  reload=True, k_fold=False, sep='random_stratified'):
    # Define directory paths for saving the processed data
    save_dir = os.path.join(os.path.dirname(dataset_file), '.'.join(os.path.basename(dataset_file).split('.')[:-1]),
                            featurizer, sep)
    overall = os.path.join(save_dir, 'overall')
    train, valid, test = os.path.join(save_dir, 'train'), os.path.join(
        save_dir, 'valid'), os.path.join(save_dir, 'test')
    if k_fold:
        test = os.path.join(save_dir, 'cv_test')
        cv = [[os.path.join(save_dir, 'cross_val', data + '_' + str(i)) for data in ['train', 'valid']] for i in
              range(k_fold)]
    header = linecache.getline(dataset_file, 1).strip('\n').split(',')
    tasks = header[1:]

    # Load existing dataset or create a new one if it does not exist
    if os.path.isdir(overall):
        if reload:
            dataset = DiskDataset(data_dir=overall)
    else:
        loader = deepchem.data.CSVLoader(tasks=tasks, smiles_field="smiles", id_field="smiles",
                                         featurizer=featurizers[featurizer])
        dataset = loader.create_dataset(dataset_file, data_dir=os.path.join(save_dir, 'overall'), shard_size=8192)

    # Apply transformations to the dataset if enabled
    if transformers:
        transformers = [deepchem.trans.BalancingTransformer(dataset=dataset)]
        for transformer in transformers:
            dataset = transformer.transform(dataset)

    # Load or split the dataset into training, validation, and testing sets
    if os.path.isdir(test):
        if reload:
            dataset = DiskDataset(data_dir=overall)
            transformers = [deepchem.trans.BalancingTransformer(dataset=dataset)]
            if not k_fold:
                dataset, train_dataset, valid_dataset, test_dataset = DiskDataset(data_dir=overall), DiskDataset(
                    data_dir=train), DiskDataset(data_dir=valid), DiskDataset(data_dir=test)
                all_dataset = (dataset, train_dataset, valid_dataset, test_dataset)
            else:
                test_dataset = DiskDataset(data_dir=test)
                cv_datasets = [(DiskDataset(train), DiskDataset(test)) for train, test in cv]
                all_dataset = (dataset, cv_datasets, test_dataset)
            return tasks, all_dataset, transformers
    else:
        splitter = splitters[sep]
        if not k_fold:
            train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
                dataset, train_dir=train, valid_dir=valid, test_dir=test, seed=123)
            all_dataset = (dataset, train_dataset, valid_dataset, test_dataset)
        else:
            cv_dataset, test_dataset = splitter.train_test_split(
                dataset, test_dir=test, frac_train=0.9, seed=123)
            cv_datasets = splitter.k_fold_split(cv_dataset, k=k_fold, directories=sum(cv, []), seed=123)
            all_dataset = (dataset, cv_datasets, test_dataset)
        return tasks, all_dataset, transformers


# Function to load datasets for withdrawal studies
def dcdata_withdraw_loader(data_file, featurizer='GraphConv', reload=True):
    save_dir = os.path.join(os.path.dirname(data_file), '.'.join(os.path.basename(data_file).split('.')[:-1]),
                            featurizer)

    header = linecache.getline(data_file, 1).strip('\n').split(',')
    tasks = header[1:]

    # Load dataset if it exists, or create a new one otherwise
    if os.path.isdir(save_dir):
        if reload:
            dataset = DiskDataset(data_dir=save_dir)
    else:
        loader = deepchem.data.CSVLoader(tasks=tasks, smiles_field="washed_smiles", id_field="washed_smiles",
                                         featurizer=featurizers[featurizer])
        dataset = loader.create_dataset(data_file, data_dir=save_dir, shard_size=8192)

    return tasks, dataset

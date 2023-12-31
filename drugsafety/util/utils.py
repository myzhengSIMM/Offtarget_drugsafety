from typing import List
import os
import numpy as np
import yaml
import json
import logging
from tqdm import tqdm as core_tqdm
from logging import Logger
from deepchem.data import NumpyDataset

def get_dcdataset_from_tasks(dcdataset, tasks):
    if not isinstance(tasks, List):
        tasks = [tasks] 
    task_indices = [np.where(dcdataset.tasks == t)[0][0] for t in tasks] 
    sample_indices = [i for i, w in enumerate(dcdataset.w[:, task_indices]) if w.any() != 0]
    X = dcdataset.X[sample_indices]
    y = dcdataset.y[sample_indices][:, task_indices] 
    w = dcdataset.w[sample_indices][:, task_indices] 
    indices = dcdataset.ids[sample_indices] 
    task_dataset = NumpyDataset(X, y, w, indices)
    return task_dataset

class tqdm(core_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ascii", True)
        super(tqdm, self).__init__(*args, **kwargs)


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        checkDirectory(save_dir)
        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def read_ckg_config(key=None):
    cwd = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(cwd, 'config/ckg_config.yml')
    content = read_yaml(config_file)
    if key is not None:
        if key in content:
            return content[key]

    return content


def save_dict_to_yaml(data, yaml_file):
    with open(yaml_file, 'w') as out:
        try:
            content = yaml.dump(data, sort_keys=False)
            out.write(content)
        except yaml.YAMLError as err:
            raise yaml.YAMLError("The yaml file {} could not be parsed. {}".format(yaml_file, err))


def read_yaml(yaml_file):
    content = None
    with open(yaml_file, 'r') as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            raise yaml.YAMLError("The yaml file {} could not be parsed. {}".format(yaml_file, err))
    return content


def get_queries(queries_file):
    queries = None
    if queries_file.endswith("yml"):
        queries = read_yaml(queries_file)
    else:
        raise Exception("The format specified in the queries file {} is not supported. {}".format(queries_file))

    return queries


def get_configuration(configuration_file):
    configuration = None
    if configuration_file.endswith("yml"):
        configuration = read_yaml(configuration_file)
    else:
        raise Exception(
            "The format specified in the configuration file {} is not supported. {}".format(configuration_file))

    return configuration


def get_configuration_variable(configuration_file, variable):
    configuration = get_configuration(configuration_file)
    if variable in configuration:
        return configuration[variable]
    else:
        raise Exception(
            "The varible {} is not found in the configuration file {}. {}".format(variable, configuration_file))


def setup_logging(path='log.config', key=None):
    """Setup logging configuration"""
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(key)

    return logger


def listDirectoryFiles(directory):
    onlyfiles = [f for f in os.listdir(directory) if
                 os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')]

    return onlyfiles


def listDirectoryFolders(directory):
    dircontent = [f for f in os.listdir(directory) if
                  os.path.isdir(os.path.join(directory, f)) and not f.startswith('.')]
    return dircontent


def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except Exception:
        return False


def convert_dash_to_json(dash_object):
    if not hasattr(dash_object, 'to_plotly_json'):
        dash_json = dash_object
    else:
        dash_json = dash_object.to_plotly_json()
        for key in dash_json:
            if isinstance(dash_json[key], dict):
                for element in dash_json[key]:
                    children = dash_json[key][element]
                    ch = {element: []}
                    if is_jsonable(children) or isinstance(children, np.ndarray):
                        ch[element] = children
                    elif isinstance(children, dict):
                        ch[element] = {}
                        for c in children:
                            ch[element].update({c: []})
                            if isinstance(children[c], list):
                                for f in children[c]:
                                    if is_jsonable(f) or isinstance(f, np.ndarray):
                                        ch[element][c].append(f)
                                    else:
                                        ch[element][c].append(convert_dash_to_json(f))
                            else:
                                if is_jsonable(children[c]) or isinstance(children[c], np.ndarray):
                                    ch[element][c] = children[c]
                                else:
                                    ch[element][c] = convert_dash_to_json(children[c])
                    elif isinstance(children, list):
                        for c in children:
                            if is_jsonable(c) or isinstance(c, np.ndarray):
                                ch[element].append(c)
                            else:
                                ch[element].append(convert_dash_to_json(c))
                    else:
                        ch[element] = convert_dash_to_json(children)
                    dash_json[key].update(ch)

    return dash_json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


class DictDFEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        return json.JSONEncoder.default(self, obj)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import best_combinations_SimpleCNN_triplet_subset as best_comb_cnn_subset
import math
import logging
import random
import argparse
from numpy.random import permutation
from sklearn import svm, datasets
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.observers import FileStorageObserver
import numpy as np
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import pickle
from math import fabs

import pymongo
import os
# MONGODB_HOST = os.environ['MONGODB_HOST']
MONGODB_HOST = 'mongo'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

##########################################
from pymongo import MongoClient
client = MongoClient(MONGODB_HOST, 27017)
db = client['sacred_db_omniboard']

#Pandas + SACRED MongoDB
#Get a dataframe that summarize the experiements, according to a PANDAS query

from collections import OrderedDict
import pandas as pd
import re

# def slice_dict(d, keys):
#     """ Returns a dictionary ordered and sliced by given keys
#         keys can be a list, or a CSV string
#     """
#     if isinstance(keys, str):
#         keys = keys[:-1] if keys[-1] == ',' else keys
#         keys = re.split(', |[, ]', keys)

#     return dict((k, d[k]) for k in keys)

# def sacred_to_df(db_runs, mongo_query=None, ):
#     """
#     db_runs is usually db.runs
#     returns a dataframe that summarizes the experiments, where 
#     config and info fields are flattened to their keys.
#     Summary DF contains the following columns:
#     _id, experiment.name, **config, result, **info, status, start_time
#     """
#     # get all experiment according to mongo query and represent as a pandas DataFrame    
#     df = pd.DataFrame(list(db_runs.find(mongo_query)))

#     # Take only the interesting columns
#     df = df.loc[:, '_id, experiment, config, result, info, status, start_time'.split(', ')]

#     def _summerize_experiment(s):
#         """
#         Take only the 
#         """
#         o = OrderedDict()
#         o['_id'] = s['_id']
#         o['name']=s['experiment']['name']
#         o.update(s['config'])
#         for key, val in s['info'].items():
#             if key != 'metrics':
#                 o[key] = val 

#         o.update(slice_dict(s.to_dict(), 'result, status, start_time'))
#         return pd.Series(o)
    
#     sum_list = []
#     for ix, s in df.iterrows():
#         sum_list.append(_summerize_experiment(s))
#     df_summary = pd.DataFrame(sum_list).set_index('_id')
    
#     return df_summary

# query = 'status=="COMPLETED"' # and val_acc>0.85 and fc_dim<=100'
# df_summary = sacred_to_df(db.runs).query(query)

##########################################

def hyperopt_objective(args_):

    # arguments to pass as config_updates dict
    global args
    # result to pass to hyperopt
    global hyperopt_result
    # command-line arguments
    global parse_args

    global all_results
    global my_metrics

    try:
        # set global args with current arguments pooled by hypeopt from parameters search space.
        args = args_

        # Create new experiment
        exp = Experiment('SimpleCNN triplet-subreg V7')
        # logger.debug("Adding observer for {}, DB {}".format(parse_args.mongo_db_address, parse_args.mongo_db_name))
        ## exp.observers.append(MongoObserver.create(
        ##     url=parse_args.mongo_db_address, db_name=parse_args.mongo_db_name))

        
        exp.observers.append(FileStorageObserver('SimpleCNN_triple-subreg_v7'))

        exp.main(run_with_global_args)
        # exp.main(inner_objective)
        # exp.add_config({'args_': args_})
        exp.add_config(args_)

        @exp.capture
        def log_metrics(_run, logs):
            _run.log_scalar("loss", float(logs.get('loss')))
            _run.log_scalar("f1", float(logs.get('f1')))
            _run.log_scalar("val_loss", float(logs.get('val_loss')))
            _run.log_scalar("val_f1", float(logs.get('val_f1')))
            _run.result = float(logs.get('val_f1'))
        my_metrics = log_metrics

        exp_run = exp.run()
        _all_results = all_results  #exp_run.result
        result_msg = "Experiment result: {}\n" "Report to hyperopt: {}".format(_all_results, hyperopt_result)
        logger.debug(result_msg)
        print(result_msg)

        # return {'loss': hyperopt_result, 'status': STATUS_OK}
        return hyperopt_result

    except Exception as e:
        print(e)
        raise e
        # # If we somehow cannot get to the MongoDB server, then continue with the experiment
        # logger.warning("Running without Sacred")
        # args = args_
        # run_with_global_args()
        # return hyperopt_result
        # # return inner_objective(args=args_)


args = None
hyperopt_result = 100.
all_results = {}
my_metrics = None

def run_with_global_args():
    global args
    global hyperopt_result
    global all_results
    global my_metrics
    try:
        # all_results - will be returned and published to Observers by Sacred Experiment object.
        all_results = best_comb_cnn_subset.objective(args=args, f_log_metrics=my_metrics)  #inner_objective(args)
        # set global hyperopt_result value - used by hypeopt to pick next parameters set.
        hyperopt_result = -1 * np.array(all_results['history']['val_f1']).sum() #-np.mean(all_results['val_f1_score'])
        return hyperopt_result
    except Exception as e:
        print(e)
        # Have Sacred log a null result
        return None


# def inner_objective(args):
#     msg = "inner_objective : {}".format(args)
#     logger.debug(msg)
#     print(msg)
#     return {'val_f1_score': random.randint(1, 101), 'val_acc': random.randint(1, 101)}


def run_hyperopt():
    space = hp.choice('attributes', [
    {
        'ch1': hp.randint('ch1_label', 25), #*10 + 450,
        'ch2': hp.randint('ch2_label', 25), #*10 + 450,
        'ch3': hp.randint('ch3_label', 25), #*10 + 450        
    }])
    trials = Trials()
    best = fmin(hyperopt_objective, space, algo=tpe.suggest,
                max_evals=int(parse_args.num_runs), trials=trials)
    print("Best run ", best)
    return trials, best

# if __name__ == '__main__':
#     ex.run_commandline()


if __name__ == '__main__':
    # """
    # Runs a series of test using Hyperopt to determine which parameters are most important
    # All tests will be logged using Sacred - if the sacred database is set up correctly, otherwise it will simply run
    # To run pass in the number of hyperopt runs, the mongo db address and name, as well as the directory of files to test
    # For example for 10 tests:
    # python experiments/hyperopt_experiments.py 10 db_server:00000 pythia data/stackexchange/anime
    # """
    # parser = argparse.ArgumentParser(
    #     description="Pythia Hyperopt Tests logging to Sacred")
    # parser.add_argument("num_runs", type=int,
    #                     help="Number of Hyperopt Runs", default=1300, nargs='?')
    # parser.add_argument("mongo_db_address", type=str, help="Address of the Mongo DB",
    #                     default='mongodb://{}:27017/'.format(MONGODB_HOST), nargs='?')
    # # default='mongodb://twino-forms:ZGL0YxnBf5QeV28KYkDdgmi6f2vYRVXjpYz8Ezc41Y6LI7TvlW9QJ7jpYQmN31ACCXdwdd2s6tUtHd0zyJv7oA==@twino-forms.documents.azure.com:10255/?ssl=true&replicaSet=globaldb', nargs='?')
    # parser.add_argument("mongo_db_name", type=str,
    #                     help="Name of the Mongo DB", default='sacred_db_omniboard', nargs='?')
    # # parser.add_argument("directory_base", type=str, help="Directory of files")

    # global parse_args
    # parse_args = parser.parse_args()

    # if int(parse_args.num_runs) <= 0:
    #     print("Must have more than one run")

    # # Let's an AutoKeras experiment on best known triplet
    # __args = {
    #     'ch1':450,
    #     'ch2':590,
    #     'ch3':950
    # }
    # hyperopt_objective(args_=__args)

    """
    Runs a series of test using Hyperopt to determine which parameters are most important
    All tests will be logged using Sacred - if the sacred database is set up correctly, otherwise it will simply run
    To run pass in the number of hyperopt runs, the mongo db address and name, as well as the directory of files to test
    For example for 10 tests:
    python experiments/hyperopt_experiments.py 10 db_server:00000 pythia data/stackexchange/anime
    """
    parser = argparse.ArgumentParser(
        description="Pythia Hyperopt Tests logging to Sacred")
    # parser.add_argument("num_runs", type=int,
    #                     help="Number of Hyperopt Runs", default=1300, nargs='?')
    parser.add_argument("mongo_db_address", type=str, help="Address of the Mongo DB",
                        default='mongodb://{}:27017/'.format(MONGODB_HOST), nargs='?')
    # default='mongodb://twino-forms:ZGL0YxnBf5QeV28KYkDdgmi6f2vYRVXjpYz8Ezc41Y6LI7TvlW9QJ7jpYQmN31ACCXdwdd2s6tUtHd0zyJv7oA==@twino-forms.documents.azure.com:10255/?ssl=true&replicaSet=globaldb', nargs='?')
    parser.add_argument("mongo_db_name", type=str,
                        help="Name of the Mongo DB", default='sacred_db_omniboard', nargs='?')
    # parser.add_argument("directory_base", type=str, help="Directory of files")

    global parse_args
    parse_args = parser.parse_args()

    # if int(parse_args.num_runs) <= 0:
    #     print("Must have more than one run")

    # Let's do a grid of exepriments
    # generate args grid
    from itertools import combinations
    from math import fabs
    L = [i for i in range(0, 51, 1)]
    triple_combs = [comb for comb in combinations(L, 3)]
    
    delta = 30

    for triple_arg in triple_combs:
        ch1 = triple_arg[0]
        ch2 = triple_arg[1]
        ch3 = triple_arg[2]

        # Best known AutoKeras wavebands:
        # 'ch1':450,
        # 'ch2':590,
        # 'ch3':950
        ch1__ = ch1*10 + 450
        ch2__ = ch2*10 + 450
        ch3__ = ch3*10 + 450
        if  (fabs(ch1__ - 450) <= delta) and \
            (fabs(ch2__ - 590) <= delta) and \
            (fabs(ch3__ - 950) <= delta):
            
            print('Experimenting with:', ch1, ch2, ch3)
            hyperopt_objective({
                'ch1':ch1,
                'ch2':triple_arg[1],
                'ch3':triple_arg[2]})
    
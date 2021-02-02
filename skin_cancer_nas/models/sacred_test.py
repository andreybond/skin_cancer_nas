"""
Trying simplify sacred_test_2.0.py to drop outer_
"""

import logging
import random
import argparse
from numpy.random import permutation
from sklearn import svm, datasets
from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import pickle

import pymongo
import os
MONGODB_HOST = os.environ['MONGODB_HOST']

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

hyperopt_estimation_counter = 1

def objective_wrapper(args_):

    # # arguments to pass as config_updates dict
    # global args
    # # result to pass to hyperopt
    # global opt_result
    # command-line arguments
    global parse_args
    # estimations counter
    global hyperopt_estimation_counter

    try:
        ex = Experiment('Hyperopt experiment #1 v1 - estimation#{}'.format(hyperopt_estimation_counter))
        hyperopt_estimation_counter = hyperopt_estimation_counter + 1
        
        logger.debug("Adding observer for {}, DB {}".format(
            parse_args.mongo_db_address, parse_args.mongo_db_name))
        ex.observers.append(MongoObserver.create(
            url=parse_args.mongo_db_address, db_name=parse_args.mongo_db_name))

        args = args_
        # ex.main(run_with_global_args)
        ex.main(inner_objective)
        ex.add_config({'args': args_})
        # ex.add_config(args_)
        r = ex.run()
        opt_result = extract_opt_param(all_results=r.result)
        result_msg = "Experiment result: {}\n" "Report to hyperopt: {}".format(r.result, opt_result)
        logger.debug(result_msg)
        print(result_msg)

    except Exception as e:
        print(e)
        # raise
        # If we somehow cannot get to the MongoDB server, then continue with the experiment
        logger.warning("Running without Sacred")
        print("Running without Sacred")
        all_results = inner_objective(args=args_)
        opt_result= extract_opt_param(all_results=all_results)

    return {'loss': opt_result, 'status': STATUS_OK}


# args = None
# opt_result = 100.

def extract_opt_param(all_results):
    try:
        opt_result = -np.mean(all_results['val_f1_score'])
        return opt_result
    except:
        # Have Sacred log a null result
        return None

def inner_objective(args):
    msg = "inner_objective : {}".format(args)
    logger.debug(msg)
    print(msg)
    return {'val_f1_score': random.randint(1, 101), 'val_acc': random.randint(1, 101)}


def run_hyperopt():
    space = {
        "algorithm_type": hp.choice('algorithm_type', [
            {
                'LOG_REG': True,
                'LOG_C': hp.choice('log_C', [1e-5, 1e-4, 1e-3, 1e-2, 1, 10]),
                'LOG_TOL': hp.choice('log_tol', [1e-5, 1e-4, 1e-3, 1e-2, 1, 10]),
                'LOG_PENALTY': hp.choice('log_penalty', ["l1", "l2"])
            }, {
                'SVM': True,
                'SVM_C': hp.choice('svm_C', [2000, 1000]),
                'SVM_KERNEL': hp.choice('svm_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'SVM_GAMMA': hp.choice('svm_gamma', ['auto', 1000, 5000, 10000])
            }])
    }
    trials = Trials()
    best = fmin(objective_wrapper, space, algo=tpe.suggest,
                max_evals=int(parse_args.num_runs), trials=trials)
    print("Best run ", best)
    return trials, best

# if __name__ == '__main__':
#     ex.run_commandline()



if __name__ == '__main__':
    """
    Runs a series of test using Hyperopt to determine which parameters are most important
    All tests will be logged using Sacred - if the sacred database is set up correctly, otherwise it will simply run
    To run pass in the number of hyperopt runs, the mongo db address and name, as well as the directory of files to test
    For example for 10 tests:
    python experiments/hyperopt_experiments.py 10 db_server:00000 pythia data/stackexchange/anime
    """
    parser = argparse.ArgumentParser(
        description="Pythia Hyperopt Tests logging to Sacred")
    parser.add_argument("num_runs", type=int,
                        help="Number of Hyperopt Runs", default=5, nargs='?')
    parser.add_argument("mongo_db_address", type=str, help="Address of the Mongo DB",
                        default='mongodb://{}:27017/'.format(MONGODB_HOST), nargs='?')
    # default='mongodb://twino-forms:ZGL0YxnBf5QeV28KYkDdgmi6f2vYRVXjpYz8Ezc41Y6LI7TvlW9QJ7jpYQmN31ACCXdwdd2s6tUtHd0zyJv7oA==@twino-forms.documents.azure.com:10255/?ssl=true&replicaSet=globaldb', nargs='?')
    parser.add_argument("mongo_db_name", type=str,
                        help="Name of the Mongo DB", default='sacred_db_omniboard', nargs='?')
    # parser.add_argument("directory_base", type=str, help="Directory of files")

    global parse_args
    parse_args = parser.parse_args()

    if int(parse_args.num_runs) <= 0:
        print("Must have more than one run")

    trial_results, best = run_hyperopt()
    print('Best result is:')
    print(best)

    # with open("sacred_hyperopt_results_test1_" + '.pkl', 'wb') as f:
    #     pickle.dump(trial_results, f, pickle.HIGHEST_PROTOCOL)
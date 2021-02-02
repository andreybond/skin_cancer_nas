from numpy.random import permutation
from sklearn import svm, datasets
from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('iris_rbf_svm')

ex.observers.append(MongoObserver.create(url='mongodb://sample:password@mongo:27017/db?authSource=admin', db_name='db'))

# mongodb://sample:password@localhost:27017/?authMechanism=SCRAM-SHA-1

@ex.config
def cfg():
  C = 1.0
  gamma = 0.7

@ex.automain
def run(C, gamma):
  iris = datasets.load_iris()
  per = permutation(iris.target.size)
  iris.data = iris.data[per]
  iris.target = iris.target[per]
  clf = svm.SVC(C, 'rbf', gamma=gamma)
  clf.fit(iris.data[:90],
          iris.target[:90])
  return clf.score(iris.data[90:],
                   iris.target[90:])
from time import time
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

#import os
#import threading

# get some data
digits = load_digits()
X, y = digits.data, digits.target

# build a classifier
clf = RandomForestClassifier(n_estimators=20, random_state=41)


# specify parameters and distributions to sample from
from skopt.space import Real, Categorical, Integer
param_dist = {"max_depth": Categorical([3, None]),
              "max_features": Integer(1, 11),
              "min_samples_split": Integer(2, 11),
              "min_samples_leaf": Integer(1, 11),
              "bootstrap": Categorical([True, False]),
              "criterion": Categorical(["gini", "entropy"])}

# callback for saving checkopoints
from skopt.callbacks import CheckpointSaver
checkpoint_callback = CheckpointSaver("./result.pkl")

# callback for monitoring results and for early termination
def montitoring_callback(res):
    current_step = len(searchcv.cv_results_['mean_test_score'])
    current_score = searchcv.cv_results_['mean_test_score'][-1]
    current_params = list(searchcv.cv_results_['params'][-1].values())

    best_step = searchcv.best_index_ + 1
    score = searchcv.best_score_
    best_params = searchcv.best_params_

    print("Step %s: Params: %s. Score: %s. Best step is %s" % 
        (current_step, current_params, current_score, best_step))

    # I used this to confirm that only one of the threads gets the callback.
    # print(threading.current_thread())

    # I used this to confirm that the pickle file is only written out if the score improves.
    # print("Last Modified: %s" % os.path.getmtime("./result.pkl"))

    if score >= 0.98:
        print('Interrupting!')
        return True

# run Bayesian search
from skopt import BayesSearchCV
n_iter_search = 20
searchcv = BayesSearchCV(clf, search_spaces=param_dist,
                                   n_iter=n_iter_search, 
                                   n_jobs=2,
                                   n_points=2,
                                   random_state=41)
start = time()
searchcv.fit(X, y, callback=[montitoring_callback, checkpoint_callback])
print("BayesSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
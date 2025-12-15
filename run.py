# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from MTGPSC import SymbolicRegressorGP
from deap import gp
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import math
import time
import random
import csv
import openml
import sys
# from scikit_mtr.multi_output_tools import load_data_by_sklearn
import arff

# random_seed=int(sys.argv[1])
# dataset=int(sys.argv[2])
dataset=41480
random_seed=25
print(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

#load dataset
with open(f"{dataset}.arff", "r") as f:
    arff_data = arff.load(f)
df = pd.DataFrame(
    arff_data["data"], columns=[attr[0] for attr in arff_data["attributes"]]
)

X = df.iloc[:, :-16].astype("float32").values
y = df.iloc[:, -16:].astype("float32").values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

start_CPU=time.process_time()
start_Wall=time.perf_counter()

# run MTGPSC
sr = SymbolicRegressorGP(n_generations=10, verbose=True)
sr.fit(X_train, y_train)
end_CPU=time.process_time()
end_Wall=time.perf_counter()




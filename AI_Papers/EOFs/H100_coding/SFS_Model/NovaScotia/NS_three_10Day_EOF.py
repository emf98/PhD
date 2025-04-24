#### Sequential Feature Selection of PCs for Random Forest model using PV/GPH/EHF as input.
## as python file ... so that I can run this without it being kicked. lol. 

#File created on April 23, 2025 and localized within the H100 cluster. 
#I would like to identify the # of nodes that represent >80% of the variance for each feature. Then, I will do two models. 

#I am changing the lead time to 14 days. 

print("SFS FOR PC MODEL OVER NOVA SCOTIA, 3 CLASSES.")
print('###########################################')
print("Start 10-day lag file ... Import statements")
# relevant import statements
import math
import pickle
import random

##just to stop the excess number of warnings
import warnings
from random import randint, sample, seed

import cartopy.feature as cf
import cartopy.util  # Requires separate import
import matplotlib.colors as mcolors

# plotting related imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # statistical data visualization
import xarray as xr
from cartopy import crs as ccrs  # Useful for plotting maps
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib import rcParams  # For changing text properties
from scipy.ndimage import gaussian_filter
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

##import tensorflow/keras related files
import tensorflow as tf
import tensorflow.keras.backend as K
from eofs.standard import Eof
from tensorflow import keras
from tensorflow.keras import (
    Input,
    Sequential,
    constraints,
    initializers,
    layers,
    optimizers,
    regularizers,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Activation,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    InputSpec,
    Layer,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# tf.compat.v1.disable_v2_behavior() # <-- HERE !


tf.compat.v1.disable_eager_execution()
import investigate
################################################
# The bulk of this file to start is copied from the previous EOF file/my EOF test file. 

##### First, I am going to pickle in the input data and solvers to reflect what was done in the original EOF file. 

#I had to actually download the data locally and upload it here to the H100 cluster in the ./Dissertation_Coding/data folder because I could not call it from my home directory. 

# load input solvers
infile = open("../../EOF_Analysis/PVsolver_10days.p","rb",)
PVsolver = pickle.load(infile)  ##pv on an isentropic surface, 350
infile.close()

infile = open("../../EOF_Analysis/EHFsolver_10days.p","rb",)
EHFsolver = pickle.load(infile)  ##ZMehf vertical cross section along longitudes
infile.close()

infile = open("../../EOF_Analysis/GPHsolver_10days.p","rb",)
GPHsolver = pickle.load(infile)  ##ZMehf vertical cross section along longitudes
infile.close()

print("Solvers loaded ...")

from EOF_def import EOF_def

## PV
PV_EOF_nw, PV_EOF_nw2d, PV_eigenv, PV_VarEx, PV_PC = EOF_def(PVsolver, 200)
## EHF
EHF_EOF_nw, EHF_EOF_nw2d, EHF_eigenv, EHF_VarEx, EHF_PC = EOF_def(EHFsolver, 55)
## GPH
GPH_EOF_nw, GPH_EOF_nw2d, GPH_eigenv, GPH_VarEx, GPH_PC = EOF_def(GPHsolver, 12)

##############################################################
# ## Now I am going to import temperature data for attempts at running a simple ANN. 

#Below is PCs combined for PV, GPH, and EHF with 14-day lagged temperatures. 1959/1960 to 2020/2021. 
print("###############################")
print("Begin SFS Model ...")

# load output data
infile = open("../../temperature_data/novatemps_anom_classedTHREE.p","rb",)
output = pickle.load(infile) 
infile.close()

inputvar = np.concatenate((PV_PC,EHF_PC,GPH_PC),axis=1) #first 150 is PV, second 40 is EHF, last 10 is GPH

##make pandas dataframe for RF
input = pd.DataFrame(inputvar)
print("Input Dataframe")
print(input)

temp = output.reshape(62, 182)
# temp = temp[:,26:177]
temp = temp[:, 31:]
temp = temp.reshape(9362,)
print("Output shape: ",temp.shape)


##Set X_all and Y_all datasets
X_all = np.copy(input)
Y_all = np.copy(temp)

##training data partition out
X_train = X_all[: 57 * 151, :]
Y_train = Y_all[: 57 * 151]

# testing data partition out
X_test = X_all[57 * 151 :, :]
Y_test = Y_all[57 * 151 :]

##set fraction of data as 5 years
frac_ind = 151 * 5

##checking my data for NaN of Infs because I need to make sure this doesn't cause
# the model to throw back no loss
if np.any(np.isnan(X_all)) or np.any(np.isinf(X_all)):
    print("NaN or Inf values found in X_all!")

if np.any(np.isnan(Y_all)) or np.any(np.isinf(Y_all)):
    print("NaN or Inf values found in Y_all!")

##############################################################
# ## Begin setting up parts of my model architecture.
# This will be the random forest intermission.
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import cross_val_score

#save PREDICTIONS
acc_reg1_val = []

acc_reg1_train = []

acc_reg1_test = []

valP = []
testP = []
trainP = []

sele_ind = []

#create initial regressor for rf to do feature selection 
print("Initalizing RF Model")
rf_reg1 = RandomForestClassifier(max_depth=5, n_estimators=300, random_state=42)
print("Starting Cross Validation ...")
##make loop for cross validation 
for l in range(0,25):
    print("Cross Val #:"+str(l))
    ##randomly choose a fraction of events for validation and training
    start = random.randrange(len(X_train[:,0])-frac_ind)
    end = start+(frac_ind)

    X_val = X_train[start:end,:]
    Y_val = Y_train[start:end]
    
    X_train1 = X_train[0:start]
    Y_train1 = Y_train[0:start]
    X_train2 = X_train[end:]
    Y_train2 = Y_train[end:]

    ##concatenate all of these
    X_tr = np.concatenate([X_train1,X_train2], axis = 0)
    Y_tr = np.concatenate((Y_train1,Y_train2))

    #train rf
    rf_reg1.fit(X_tr, Y_tr)

    #prediction with validation data
    pred_val = rf_reg1.predict(X_val)
    valP.append(pred_val)
    acc_reg1_val.append(accuracy_score(Y_val, pred_val))
    
    #prediction with training data
    pred_train = rf_reg1.predict(X_tr)
    trainP.append(pred_train)
    acc_reg1_train.append(accuracy_score(Y_tr, pred_train))

    #prediction with testing data
    pred_test = rf_reg1.predict(X_test)
    testP.append(pred_test)
    acc_reg1_test.append(accuracy_score(Y_test, pred_test))

    #prepare to show relevant features by actually ... choosing them
    sfs = SFS(rf_reg1, 
              n_features_to_select=8, 
              direction='forward', 
              scoring='r2',
              cv=3,
              n_jobs=-1)
    print("After SFS")
    sfs = sfs.fit(X_tr, Y_tr)
    print("After fit")
    sele_ind.append(np.arange(X_tr.shape[1])[sfs.support_])
    X_train_sele = sfs.transform(X_tr)
    X_test_sele = sfs.transform(X_test)
    rf_reg_sele = rf_reg1.fit(X_train_sele,Y_tr)
    #weights_sele[ii,:] = model_linear_s.coef_
    print("Selected Features: ",np.arange(X_tr.shape[1])[sfs.support_])

print('###################################################')
print("Model CV Completed.")
print(f'Accuracy, Validation: {np.mean(acc_reg1_val) * 100:.2f}%')
print(f'Accuracy, Training: {np.mean(acc_reg1_train) * 100:.2f}%')
print(f'Accuracy, Testing: {np.mean(acc_reg1_test) * 100:.2f}%')

#pickle out the solver
print("Pickle Out Selected Indices")
pickle.dump(sele_ind, open("Nova3_SelectIndices_10days.p", 'wb'))

##150 up to is PV
##150-190 is EHF
##190-200 is GPH
print("Length of selected features array: ",len(sele_ind))
print("Selected Features: ", sele_ind)

##visualize the distribution of selected features
plt.title("Frequency of Selected PCs for 25 CVs of RF Model for Nova Scotia (10 Days, 3 Classes)")
sele_ind1 = np.array(sele_ind)
sele_ind1 = sele_ind1.reshape(25*8)
plt.hist(sele_ind1, bins=np.arange(200))
plt.savefig('../images/Nova3_10days_selefeature_hist.png')


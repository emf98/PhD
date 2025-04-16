#### Sequential Feature Selection of PCs for Random Forest model using PV/GPH/EHF as input.
## as python file ... so that I can run this without it being kicked. lol. 

#File created on April 14, 2025 and localized within the H100 cluster. 
#I would like to identify the # of nodes that represent >80% of the variance for each feature. Then, I will do two models. 

#I am changing the lead time to 10 days. 
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

##### First, I am going to pickle in the input data. I will then remove the seasonal climo from the dataset...

#I had to actually download the data locally and upload it here to the H100 cluster in the ./Dissertation_Coding/data folder because I could not call it from my home directory. 

# load input data
print("Load input data ...")
infile = open("../../data/cap_pv350pt.p","rb",)
pv_input = pickle.load(infile)  ##pv on an isentropic surface, 350
infile.close()

infile = open("../../data/1959ZMeddyheatflux.p","rb",)
ehf_input = pickle.load(infile)  ##ZMehf vertical cross section along longitudes
infile.close()

infile = open("../../data/1959gph.p","rb",)
gph_input = pickle.load(infile)  ##ZMehf vertical cross section along longitudes
infile.close()

#remove extra pv and gph year
pv_input = np.delete(pv_input, [62], 0)

## remove leap day.
pv_input = np.delete(pv_input, [151], 1)
ehf_input = np.delete(ehf_input, [151], 1)
gph_input = np.delete(gph_input, [151], 1)

print("PV Shape: ",pv_input.shape) ##62 years, october through march, 16 lats (90,60), all lons. 
print("GPH Shape: ",gph_input.shape)
print("EHF Shape: ",ehf_input.shape)

##take seasonal daily average/remove seasonal climatology
dailymean_pv = np.nanmean(pv_input, axis=1)
anom_pv = np.zeros_like(pv_input)
for t in np.arange(pv_input.shape[1]):
    anom_pv[:, t, :, :] = pv_input[:, t, :, :] - dailymean_pv
print("ANOM PV shape: ",anom_pv.shape)

dailymean_ehf = np.nanmean(ehf_input, axis=1)
anom_ehf = np.zeros_like(ehf_input)
for t in np.arange(ehf_input.shape[1]):
    anom_ehf[:, t, :, :] = ehf_input[:, t, :, :] - dailymean_ehf
print("ANOM EHF shape: ",anom_ehf.shape)

dailymean_gph = np.nanmean(gph_input, axis=1)
anom_gph = np.zeros_like(gph_input)
for t in np.arange(gph_input.shape[1]):
    anom_gph[:, t, :, :] = gph_input[:, t, :, :] - dailymean_gph
print("ANOM GPH shape: ",anom_gph.shape)

##check for NaNs
if np.any(np.isnan(anom_pv)) or np.any(np.isinf(anom_pv)):
    print("NaN or Inf values found in PV!")

if np.any(np.isnan(anom_ehf)) or np.any(np.isinf(anom_ehf)):
    print("NaN or Inf values found in EHF!")

if np.any(np.isnan(anom_gph)) or np.any(np.isinf(anom_gph)):
    print("NaN or Inf values found in GPH!")

##flatten array
print("Flattened array shapes ...")
flat_anom_pv = anom_pv[:62, 21:172, :, :].reshape((62 * 151, 16, 180))
print(flat_anom_pv.shape)

flat_EHFanom = anom_ehf[:62, 21:172, :, :].reshape((62 * 151, 37, 180))
print(flat_EHFanom.shape)

flat_GPHanom = anom_gph[:62, 21:172, :, :].reshape((62 * 151, 37, 180))
print(flat_GPHanom.shape)

#### Great, so I have everything uploaded and reduced to daily anomalies with the seasonal climo removed. Fantastic, lol. 

#Now I need to weight PV appropriately. lol. 
#I do not need to do this for EHF or GPH since I already reduced them. 

##I DID NOT USE THIS FOR THE EOFs, JUST TRIED TO CALCULATE HERE...
flat_anom = xr.DataArray(
    data=flat_anom_pv,
    dims=["dates", "lat", "lon"],
    coords=dict(
        dates=range(0, 9362, 1), lat=np.arange(90, 58, -2), lon=np.arange(0, 360, 2)
    ),
)
wgts = np.cos(flat_anom.lat / 180 * np.pi) ** 0.5
flat_PVanom = flat_anom * wgts
print("Flattened, weighted PV shape: ",flat_PVanom.values.shape)

##### Attempt at EOF analysis/PC decomp based on Zheng's example code. 

# EOF, the code above had selection of certain pressure levels and latitude bands. I do not need to do that here.
print("###############################")
print("Conducting EOF Analysis ...")
PVsolver = Eof(flat_PVanom.values)
EHFsolver = Eof(flat_EHFanom)
GPHsolver = Eof(flat_GPHanom)

#from previous files ...
#### PV = 100 PCs. 
#### EHF = 70 PCs. 
#### GPH = 25 PCs. 

#Now run EOF analysis with the "correct" inital number of modes.

#pickle out the solver
print("Pickle Out Solvers ...")
pickle.dump(PVsolver, open("PVsolver_10days.p", 'wb'))
pickle.dump(EHFsolver, open("EHFsolver_10days.p", 'wb'))
pickle.dump(GPHsolver, open("GPHsolver_10days.p", 'wb'))

from EOF_def import EOF_def

## PV
PV_EOF_nw, PV_EOF_nw2d, PV_eigenv, PV_VarEx, PV_PC = EOF_def(PVsolver, 100)
## EHF
EHF_EOF_nw, EHF_EOF_nw2d, EHF_eigenv, EHF_VarEx, EHF_PC = EOF_def(EHFsolver, 70)
## GPH
GPH_EOF_nw, GPH_EOF_nw2d, GPH_eigenv, GPH_VarEx, GPH_PC = EOF_def(GPHsolver, 25)

##############################################################
# ## Now I am going to import temperature data for attempts at running a simple ANN. 

#Below is 200 PCs combined for PV, GPH, and EHF with 14-day lagged temperatures. 1959/1960 to 2020/2021. 
print("###############################")
print("Begin SFS Model ...")

# load output data
infile = open("../../data/classed_europetemps_median.p","rb",)
output = pickle.load(infile)  ##pv on an isentropic surface, 350
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
mse_reg1_val = []
r2_reg1_val = []
acc_reg1_val = []

mse_reg1_train = []
r2_reg1_train = []
acc_reg1_train = []

mse_reg1_test = []
r2_reg1_test = []
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
    mse_reg1_val.append(mean_squared_error(Y_val, pred_val))
    r2_reg1_val.append(r2_score(Y_val, pred_val))
    acc_reg1_val.append(accuracy_score(Y_val, pred_val))
    
    #prediction with training data
    pred_train = rf_reg1.predict(X_tr)
    trainP.append(pred_train)
    mse_reg1_train.append(mean_squared_error(Y_tr, pred_train))
    r2_reg1_train.append(r2_score(Y_tr, pred_train))
    acc_reg1_train.append(accuracy_score(Y_tr, pred_train))

    #prediction with testing data
    pred_test = rf_reg1.predict(X_test)
    testP.append(pred_test)
    mse_reg1_test.append(mean_squared_error(Y_test, pred_test))
    r2_reg1_test.append(r2_score(Y_test, pred_test))
    acc_reg1_test.append(accuracy_score(Y_test, pred_test))

    #prepare to show relevant features by actually ... choosing them
    sfs = SFS(rf_reg1, 
              n_features_to_select=14, 
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
pickle.dump(sele_ind, open("SelectIndices14_10days.p", 'wb'))

##100 up to is PV
##100-170 is EHF
##170-195 is GPH
print("Length of selected features array: ",len(sele_ind))
print("Selected Features: ", sele_ind)

##visualize the distribution of selected features
plt.title("Frequency of Selected PCs for 25 CVs of RF Model")
sele_ind1 = np.array(sele_ind)
sele_ind1 = sele_ind1.reshape(25*14)
plt.hist(sele_ind1, bins=np.arange(195))
plt.savefig('10days_sele_14feature_hist.png')


##definition statements for statistics 
import sklearn
from sklearn.metrics import brier_score_loss
import xarray as xr
import numpy as np

###################################################
##Brier Skill Scores
def BSS(iter, i, days,
        climo_full, pred, Y_all, BSS_all,
        climo_val, pred_val, Y_validation, BSS_val,
        climo_train, pred_train, Y_train, BSS_train,
        climo_test, pred_test, Y_test, BSS_test):
    
    for j in range(len(climo_full[0,:])):
        bs = brier_score_loss(Y_all[:,j], pred[:,j])
        bs_ref = brier_score_loss(Y_all[:,j], climo_full[:,j])
        bss = 1 - (bs / bs_ref)
            
        BSS_all[iter,j] = round(bss,4)
            
    for j in range(len(climo_val[0,:])):
        bs = brier_score_loss(Y_validation[:,j], pred_val[:,j])
        bs_ref = brier_score_loss(Y_validation[:,j], climo_val[:,j])
        bss = 1 - (bs / bs_ref)
            
        BSS_val[iter,j] = round(bss,4)
            
    for j in range(len(climo_train[0,:])):
        bs = brier_score_loss(Y_train[:,j], pred_train[:,j])
        bs_ref = brier_score_loss(Y_train[:,j], climo_train[:,j])
        bss = 1 - (bs / bs_ref)
            
        BSS_train[iter,j] = round(bss,4)

    for j in range(len(climo_test[0,:])):
        bs = brier_score_loss(Y_test[:,j], pred_test[:,j])
        bs_ref = brier_score_loss(Y_test[:,j], climo_test[:,j])
        bss = 1 - (bs / bs_ref)
            
        BSS_test[iter,j] = round(bss,4)
    
    return BSS_all, BSS_val, BSS_train, BSS_test;

###################################################
##Recall Accuracy Scores
def RAS_two(iter, Rec_all, climo_full, Y_all, pred, pred_class,
            climo_val, Rec_val, Y_validation, pred_val, predval_class,
            climo_train, Rec_train, Y_train, pred_train, predtr_class,
            climo_test, Rec_test, Y_test, pred_test, predtest_class):
    
    pred_class = []
    predval_class = []
    predtr_class = []
    predtest_class = []
    
    for k in range(len(Y_all[:,0])):##PREDICTION
        #print(pred_val[k,0])
        #print(pred_val[k,1])
        #print(pred_val[k,2])
        if pred[k,0] > pred[k,1]:
            #print("a")
            pred_class.append([1.,0.])
        elif pred[k,1] > pred[k,0]:
            #print("b")
            pred_class.append([0.,1.])
    prd = np.asarray(pred_class)
    for j in range(len(climo_full[0,:])):
        recal = sklearn.metrics.recall_score(Y_all[:,j], prd[:,j])
        Rec_all[iter,j] = round(recal,4)
    
    for k in range(len(Y_validation[:,0])): ##VALIDATION
        #print(pred_val[k,0])
        #print(pred_val[k,1])
        #print(pred_val[k,2])
        if pred_val[k,0] > pred_val[k,1]:
            #print("a")
            predval_class.append([1.,0.])
        elif pred_val[k,1] > pred_val[k,0]:
            #print("b")
            predval_class.append([0.,1.])
    val = np.asarray(predval_class)
    for j in range(len(climo_val[0,:])):
        recal = sklearn.metrics.recall_score(Y_validation[:,j], val[:,j])
        Rec_val[iter,j] = round(recal,4)
            
    for k in range(len(Y_train[:,0])): ##TRAINING
        #print(pred_val[k,0])
        #print(pred_val[k,1])
        #print(pred_val[k,2])
        if pred_train[k,0] > pred_train[k,1]:
            #print("a")
            predtr_class.append([1.,0.])
        elif pred_train[k,1] > pred_train[k,0]:
            #print("b")
            predtr_class.append([0.,1.])
    tri = np.asarray(predtr_class)
    for j in range(len(climo_train[0,:])):
        recal = sklearn.metrics.recall_score(Y_train[:,j], tri[:,j])
        Rec_train[iter,j] = round(recal,4)

    for k in range(len(Y_test[:,0])): ##TESTING
        #print(pred_val[k,0])
        #print(pred_val[k,1])
        #print(pred_val[k,2])
        if pred_test[k,0] > pred_test[k,1]:
            #print("a")
            predtest_class.append([1.,0.])
        elif pred_test[k,1] > pred_test[k,0]:
            #print("b")
            predtest_class.append([0.,1.])
    tes = np.asarray(predtest_class)
    for j in range(len(climo_test[0,:])):
        recal = sklearn.metrics.recall_score(Y_test[:,j], tes[:,j])
        Rec_test[iter,j] = round(recal,4)
    
    return Rec_all, Rec_val, Rec_train, Rec_test;


###################################################
##Precision Accuracy Scores
def PAS_two(iter, Prec_all, climo_full, Y_all, pred, pred_class,
            climo_val, Prec_val, Y_validation, pred_val, predval_class,
            climo_train, Prec_train, Y_train, pred_train, predtr_class,
            climo_test, Prec_test, Y_test, pred_test, predtest_class):
    
    pred_class = []
    predval_class = []
    predtr_class = []
    predtest_class = []
    
    for k in range(len(Y_all[:,0])):##PREDICTIVE
        #print(pred_val[k,0])
        #print(pred_val[k,1])
        #print(pred_val[k,2])
        if pred[k,0] > pred[k,1]:
            #print("a")
            pred_class.append([1.,0.])
        elif pred[k,1] > pred[k,0]:
            #print("b")
            pred_class.append([0.,1.])
    prd = np.asarray(pred_class)
    for j in range(len(climo_full[0,:])):
        prec = sklearn.metrics.precision_score(Y_all[:,j], prd[:,j])
        Prec_all[iter,j] = round(prec,4)
    
    for k in range(len(Y_validation[:,0])): ##VALIDATION
        #print(pred_val[k,0])
        #print(pred_val[k,1])
        #print(pred_val[k,2])
        if pred_val[k,0] > pred_val[k,1]:
            #print("a")
            predval_class.append([1.,0.])
        elif pred_val[k,1] > pred_val[k,0]:
            #print("b")
            predval_class.append([0.,1.])
    val = np.asarray(predval_class)
    for j in range(len(climo_val[0,:])):
        prec = sklearn.metrics.precision_score(Y_validation[:,j], val[:,j])
        Prec_val[iter,j] = round(prec,4)
            
    for k in range(len(Y_train[:,0])): ##TRAINING
        #print(pred_val[k,0])
        #print(pred_val[k,1])
        #print(pred_val[k,2])
        if pred_train[k,0] > pred_train[k,1]:
            #print("a")
            predtr_class.append([1.,0.])
        elif pred_train[k,1] > pred_train[k,0]:
            #print("b")
            predtr_class.append([0.,1.])
    tri = np.asarray(predtr_class)
    for j in range(len(climo_train[0,:])):
        prec = sklearn.metrics.precision_score(Y_train[:,j], tri[:,j])
        Prec_train[iter,j] = round(prec,4)

    for k in range(len(Y_test[:,0])): ##TESTING
        #print(pred_val[k,0])
        #print(pred_val[k,1])
        #print(pred_val[k,2])
        if pred_test[k,0] > pred_test[k,1]:
            #print("a")
            predtest_class.append([1.,0.])
        elif pred_test[k,1] > pred_test[k,0]:
            #print("b")
            predtest_class.append([0.,1.])
    tes = np.asarray(predtest_class)
    for j in range(len(climo_test[0,:])):
        prec = sklearn.metrics.precision_score(Y_test[:,j], tes[:,j])
        Prec_test[iter,j] = round(prec,4)

    return Prec_all, Prec_val, Prec_train, Prec_test;

import numpy as np
import xarray as xr
##create definition statement for creating climatology

def climo(dataset,yeardim,daydim):
    #reshape data to (year,day)
    decode = np.argmax(dataset, axis=-1) #remove one hot encoding, briefly 
    temp = decode.reshape(yeardim,daydim)
    ##create empty list
    daily_t = np.empty((daydim,2))
    
    ##daily probability of each category per area
    for i in range(0,daydim): ##loop days
        day0 = []
        day1 = []
    
        for j in range(0,yeardim): ##loop years
            if temp[j,i] == 0:
                day0.append(1)
            if temp[j,i] == 1:
                day1.append(1)
        
            prob0 = len(day0)/yeardim
            #print(prob0)
            prob1 = len(day1)/yeardim
            #list = [prob0,prob1,prob2,prob3]
            list = [prob0,prob1]
        
            daily_t[i,:] = list

    #reiterate per number of years
    full_t = np.empty((yeardim, daydim,2)) ##empty dataset for iterating years
    for i in range(0,yeardim):
        full_t[i] = daily_t ##append daily to dataset

    flat = yeardim*daydim ##indicate flattened dimensions
    ft = full_t.reshape((flat,2)) #reshape to (flat, 2) to match one-hot encoded variables

    return ft;
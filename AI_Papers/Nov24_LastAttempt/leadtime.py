## wee woo wee woo this is a file for definition statements on lead time because I am tired of these taking up excess space in my code. 

##0-day; 
#range_low = 10
#range_high = 181
#leadtop = 10
#leadbott = 0

##5-day; 
#range_low = 15
#range_high = 181
#leadtop = 15
#leadbott = 5

##10-day; 
#range_low = 20
#range_high = 181
#leadtop = 20
#leadbott = 10

##14-day; 
#range_low = 24
#range_high = 181
#leadtop = 24
#leadbott = 14

##20-day; 
#range_low = 30
#range_high = 181
#leadtop = 30
#leadbott = 20

import numpy as np

def leadtime_3classif(t_c,range_low,range_high,leadtop,leadbott,
                     c_q30, c_q70, c_mx, w,sn,ct,eh,cs,tc,clim,
                     wind,size,cenlat,ehf100,classif,climo):
    for i in range(len(t_c[:,0])):
        for j in range(range_low,range_high,1):
            if t_c[i,j] <= c_q30:
                if np.any(t_c[i,j-leadtop:j-leadbott] <= c_q30):
                    #print("boo")
                    continue
                else: 
                    #print(label[i,j])
                    w.append(wind[i,j-leadtop:j-leadbott])
                    sn.append(size[i,j-leadtop:j-leadbott])
                    ct.append(cenlat[i,j-leadtop:j-leadbott])
                    eh.append(ehf100[i,j-leadtop:j-leadbott])
                    cs.append(classif[i,j-leadtop:j-leadbott])
                
                    tc.append(0)
                    #clim.append([climo[i,j,0],climo[i,j,2]])
                    clim.append(climo[i,j,:])
                
                    
            if t_c[i,j] >= c_q70 and t_c[i,j] <= c_mx:
                if np.any(t_c[i,j-leadtop:j-leadbott] >= c_q70):
                    #print("boo")
                    continue
                else: 
                    #print(label[i,j])
                    w.append(wind[i,j-leadtop:j-leadbott])
                    sn.append(size[i,j-leadtop:j-leadbott])
                    ct.append(cenlat[i,j-leadtop:j-leadbott])
                    eh.append(ehf100[i,j-leadtop:j-leadbott])
                    cs.append(classif[i,j-leadtop:j-leadbott])
                    
                    tc.append(2) ### <--- CHANGE WHEN NOT 2 CAT
                    #clim.append([climo[i,j,0],climo[i,j,2]])
                    clim.append(climo[i,j,:])
                
            if t_c[i,j] > c_q30 and t_c[i,j] < c_q70:
                if np.any(t_c[i,j-leadtop:j-leadbott] > c_q30) and np.any(t_c[i,j-leadtop:j-leadbott] < c_q70):
                    #print("boo")
                    continue
                else: 
                    w.append(wind[i,j-leadtop:j-leadbott])
                    sn.append(size[i,j-leadtop:j-leadbott])
                    ct.append(cenlat[i,j-leadtop:j-leadbott])
                    eh.append(ehf100[i,j-leadtop:j-leadbott])
                    cs.append(classif[i,j-leadtop:j-leadbott])
                    
                    tc.append(1)
                    ##clim.append([climo[i,j,0],climo[i,j,2]])
                    clim.append(climo[i,j,:])
    return tc, clim;

########################################################################################################################

def leadtime_2classif(t_c,range_low,range_high,leadtop,leadbott,
                     c_q30, c_q70, c_mx, w,sn,ct,eh,cs,tc,clim,
                     wind,size,cenlat,ehf100,classif,climo):
    for i in range(len(t_c[:,0])):
        for j in range(range_low,range_high,1):
            if t_c[i,j] <= c_q30:
                if np.any(t_c[i,j-leadtop:j-leadbott] < c_q30):
                    #print("boo")
                    continue
                else:
                    if np.any(t_c[i,j:j-2] < c_q30):
                        continue
                    else:
                        #print(label[i,j])
                        w.append(wind[i,j-leadtop:j-leadbott])
                        sn.append(size[i,j-leadtop:j-leadbott])
                        ct.append(cenlat[i,j-leadtop:j-leadbott])
                        eh.append(ehf100[i,j-leadtop:j-leadbott])
                        cs.append(classif[i,j-leadtop:j-leadbott])
                    
                        tc.append(0)
                        clim.append([climo[i,j,0],climo[i,j,1]])
                
                    
            if t_c[i,j] > c_q70 and t_c[i,j] <= c_mx:
                if np.any(t_c[i,j-leadtop:j-leadbott] > c_q70):
                    #print("boo")
                    continue
                else:
                    if np.any(t_c[i,j:j-2] > c_q70):
                        continue
                    else:
                        #print(label[i,j])
                        w.append(wind[i,j-leadtop:j-leadbott])
                        sn.append(size[i,j-leadtop:j-leadbott])
                        ct.append(cenlat[i,j-leadtop:j-leadbott])
                        eh.append(ehf100[i,j-leadtop:j-leadbott])
                        cs.append(classif[i,j-leadtop:j-leadbott])
                    
                        tc.append(1)
                        clim.append([climo[i,j,0],climo[i,j,1]])
                
    return tc, clim;
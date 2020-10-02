#################################################################
# Marjola Thanaj, PhD -  Imperial College London, 15/09/2020    #
#################################################################

import os
import numpy as np  
from numpy import array
import pandas as pd
import math
import glob
#from   tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# Theta in degrees (Optional)
def cart2cylc(x, y, z):
       r = np.sqrt(np.power(x,2)+np.power(y,2))
       t = math.atan(y/x)*(180/math.pi)
       z = z
       coord = [r,t,z]
       return coord

def unitvar(x, y, z):
       u = (1/math.sqrt(3))*(x+y+z)
       v = (1/math.sqrt(6))*(x+y-(2*z))
       w = (1/math.sqrt(2))*(x-y)
       coord = [u, v, w]
       return coord


path_data    = "/mnt/storage/home/mthanaj/cardiac/UKBB_40616/UKBB_test/4DSegment2.0_test_motion_final"
folder = os.listdir(path_data)
for iP in range(0,10):
       file = os.path.join(os.path.join(path_data,folder[iP],"motion"))
       os.chdir(file)
       txt_files = array(glob.glob("*.txt"))
       files = txt_files[0:100]
       ir=1
       npo = 50656
       Sradial = np.zeros((npo,50))
       Scirc = np.zeros((npo,50))
       Slong = np.zeros((npo,50))
       for iF in range(0,50):
              # Step 1 - Call epi and endo, project orthogonally onto the unit variable and bind
              os.chdir(file)
              print(file)
              EDendo = pd.read_csv(files[0,], sep=" ", header=None)
              EDendo.columns = ["x", "y", "z"]
              EDepi = pd.read_csv(files[50,], sep=" ", header=None)
              EDepi.columns = ["x", "y", "z"]
              EDepi_data = array(unitvar(EDepi.iloc[:,0],EDepi.iloc[:,1],EDepi.iloc[:,2])).T
              EDendo_data = array(unitvar(EDendo.iloc[:,0],EDendo.iloc[:,1],EDendo.iloc[:,2])).T
              ED_data = np.concatenate([EDepi_data, EDendo_data], axis=0)
              ED_data= pd.DataFrame(ED_data, columns=["x", "y", "z"])
              ESendo = pd.read_csv(files[iF,], sep=" ", header=None)
              ESendo.columns = ["x", "y", "z"]
              ESepi = pd.read_csv(files[iF+50,], sep=" ", header=None)
              ESepi.columns = ["x", "y", "z"]
              ESepi_data = array(unitvar(ESepi.iloc[:,0],ESepi.iloc[:,1],ESepi.iloc[:,2])).T
              ESendo_data = array(unitvar(ESendo.iloc[:,0],ESendo.iloc[:,1],ESendo.iloc[:,2])).T
              ES_data = np.concatenate([ESepi_data, ESendo_data], axis=0)
              ES_data= pd.DataFrame(ES_data, columns=["x", "y", "z"])
              path_strain = "/mnt/storage/home/mthanaj/cardiac/Experiments_of_Maria/3Dstrain_analysis"
              file_strain = os.path.join(os.path.join(path_strain,folder[iP]))
              os.chdir(file_strain)
              print(file_strain)
              # Step 2 - Transform ED_all from cartesian to cylindrical coordinates (Optional)
              ED_datan = np.zeros((len(ED_data.iloc[:,0]),3))
              ES_datan = np.zeros((len(ES_data.iloc[:,0]),3))
              for iE in range (0,(len(ED_data.iloc[:,0]))):
                     ED_datan[iE,:] = array(cart2cylc(ED_data.iloc[iE,0],ED_data.iloc[iE,1],ED_data.iloc[iE,2]))
                     ES_datan[iE,:] = array(cart2cylc(ES_data.iloc[iE,0],ES_data.iloc[iE,1],ES_data.iloc[iE,2]))
                     continue
              ED_epi = ED_datan[0:(len(EDepi_data[:,0])),:]
              ED_endo = ED_datan[(len(EDepi_data[:,0])):(len(ED_data.iloc[:,0])),:]
              ES_epi = ES_datan[0:(len(ESepi_data[:,0])),:]
              ES_endo = ES_datan[(len(ESepi_data[:,0])):(len(ES_data.iloc[:,0])),:]
              """
              # or use cartesian coordinates
              ED_datan = ED_data
              ES_datan = ES_data
              ED_epi = EDepi_data
              ED_endo = EDendo_data
              ES_epi = ESepi_data
              ES_endo = ESendo_data
              """
              # Step 3 - Compute 1% of all data points in LV and find the 1% knn in ED and ES surface get the knns 
              sc_1 = round(1*len(ED_epi[:,0])/100)
              # find the sc_1 knn in surface for ED and ES
              nPoints = np.arange(len(ED_datan.iloc[:,0]))
              nbrs1 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(ED_endo, ED_epi)
              distances_ed, con_ed = nbrs1.kneighbors(ED_epi)
              nbrs2 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(ES_endo, ES_epi)
              distances_es, con_es = nbrs2.kneighbors(ES_epi)
              X3 = ED_epi[:,1]
              X3 = X3.reshape(-1,1)
              nbrs3 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(X3)
              distances_ed2, con_ed2 = nbrs3.kneighbors(X3)
              X4 = ES_epi[:,1]
              X4 = X4.reshape(-1,1)
              nbrs4 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(X4)
              distances_es2, con_es2 = nbrs4.kneighbors(X4)
              X5 = ED_epi[:,2]
              X5 = X5.reshape(-1,1)
              nbrs5 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(X5)
              distances_ed3, con_ed3 = nbrs5.kneighbors(X5)
              X6 = ES_epi[:,2]
              X6 = X6.reshape(-1,1)
              nbrs6 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(X6)
              distances_es3, con_es3 = nbrs6.kneighbors(X6)
              nbrs11 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(ED_epi, ED_endo)
              distances_edd, con_edd = nbrs11.kneighbors(ED_endo)
              nbrs21 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(ES_epi, ES_endo)
              distances_esd, con_esd = nbrs21.kneighbors(ES_endo)
              X31 = ED_endo[:,1]
              X31 = X31.reshape(-1,1)
              nbrs31 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(X31)
              distances_edd2, con_edd2 = nbrs31.kneighbors(X31)
              X41 = ES_endo[:,1]
              X41 = X41.reshape(-1,1)
              nbrs41 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(X41)
              distances_esd2, con_esd2 = nbrs41.kneighbors(X41)
              X51 = ED_endo[:,2]
              X51 = X51.reshape(-1,1)
              nbrs51 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(X51)
              distances_edd3, con_edd3 = nbrs51.kneighbors(X51)
              X61 = ES_endo[:,2]
              X61 = X61.reshape(-1,1)
              nbrs61 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(X61)
              distances_esd3, con_esd3 = nbrs61.kneighbors(X61)
              # Step 4 - Compute strain
              Srepi = (np.sum(distances_es, axis=1)-np.sum(distances_ed, axis=1))/np.sum(distances_ed, axis=1)
              Srendo = (np.sum(distances_esd, axis=1)-np.sum(distances_edd, axis=1))/np.sum(distances_edd, axis=1)
              Scepi = (np.sum(distances_es2, axis=1)-np.sum(distances_ed2, axis=1))/np.sum(distances_ed2, axis=1)
              Scendo = (np.sum(distances_esd2, axis=1)-np.sum(distances_edd2, axis=1))/np.sum(distances_edd2, axis=1)
              Slepi = (np.sum(distances_es3, axis=1)-np.sum(distances_ed3, axis=1))/np.sum(distances_ed3, axis=1)
              Slendo = (np.sum(distances_esd3, axis=1)-np.sum(distances_edd3, axis=1))/np.sum(distances_edd3, axis=1)
              Sradial[:,iF] = np.concatenate([Srepi,Srendo], axis=0)
              Scirc[:,iF] = np.concatenate([Scepi,Scendo], axis=0)
              Slong[:,iF] = np.concatenate([Slepi,Slendo], axis=0)
              print(ir)
              ir+=1
              continue
       Sradial = pd.DataFrame(Sradial)
       Sradial.to_csv(os.path.join("tag_atlas/Sradial.txt"), index=False, header=False)
       Scirc = pd.DataFrame(Scirc)
       Scirc.to_csv(os.path.join("tag_atlas/Scirc.txt"), index=False, header=False)
       Slong = pd.DataFrame(Slong)
       Slong.to_csv(os.path.join("tag_atlas/Slong.txt"), index=False, header=False)
       continue

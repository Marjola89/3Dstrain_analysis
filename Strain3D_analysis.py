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
#import matplotlib.pyplot as plt

# Theta in degrees
def cart2cylc(x, y, z):
       r = np.sqrt(np.power(x,2)+np.power(y,2))
       t = math.atan(y/x)*(180/math.pi)
       z = z
       coord = [r,t,z]
       return coord

# This function is optional
# Theta in degrees
def cart2sph(x, y, z):
       r = math.sqrt(np.power(x,2)+np.power(y,2)+np.power(z,2))
       t = math.acos(z/r)*(180/math.pi)
       p = math.atan(y/x)*(180/math.pi)
       coord = [r,t,p]
       return coord

def unitvar(x, y, z):
       u = (1/math.sqrt(3))*(x+y+z)
       v = (1/math.sqrt(6))*(x+y-(2*z))
       w = (1/math.sqrt(2))*(x-y)
       coord = [u, v, w]
       return coord

# Strain computation
def etens(m_cyl):
       b = m_cyl[:,:3].T
       a = m_cyl[:,3:6].T
       X = np.dot(b,b.T)
       X = np.linalg.pinv(X, rcond = 1e-21)
       FF = np.dot(a,np.dot(b.T,X))
       ident = np.eye(3)
       E_L = 0.5*((np.dot(FF.T,FF))-ident) # Lagrangian strain
       E_E = 0.5*(ident-(np.dot(np.linalg.pinv(FF.T),np.linalg.pinv(FF))))
       E_L = E_L.flatten()
       E_E = E_E.flatten()
       strain = np.concatenate([E_L, E_E], axis=0)
       return strain

path_data    = "/mnt/storage/home/mthanaj/cardiac/UKBB_40616/UKBB_test/4DSegment2.0_test_motion_final"
folder = os.listdir(path_data)
for iP in range(0,10):
       file = os.path.join(os.path.join(path_data,folder[iP],"motion"))
       os.chdir(file)
       txt_files = array(glob.glob("*.txt"))
       files = txt_files[0:100]
       ir=1
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
              path_strain = "/mnt/storage/home/mthanaj/cardiac/Experiments_of_Maria/strain_analysis"
              file_strain = os.path.join(os.path.join(path_strain,folder[iP]))
              os.chdir(file_strain)
              print(file_strain)
              # Step 2 - Find ~50 knn in epi that match with endo for both ED and ES
              nbrs1 = NearestNeighbors(n_neighbors=50, algorithm='auto').fit(EDendo_data, EDepi_data) # to check!!
              distances_ed, con_ed_epi = nbrs1.kneighbors(EDepi_data)
              nbrs2 = NearestNeighbors(n_neighbors=50, algorithm='auto').fit(ESendo_data, ESepi_data)
              distances_es, con_es_epi = nbrs2.kneighbors(ESepi_data)
              print(files[iF,])
              # Step 3 - Compute middle surface
              mid_ed = (EDendo_data[con_ed_epi[:,0],:]+EDepi_data)/2
              mid_es = (ESendo_data[con_es_epi[:,0],:]+ESepi_data)/2
              for iEx in range(0,49):
                     mid_ed = (mid_ed+(EDendo_data[con_ed_epi[:,iEx],:]+EDepi_data)/2)/2 # to check!!
                     mid_es = (mid_es+(ESendo_data[con_es_epi[:,iEx],:]+ESepi_data)/2)/2
                     continue
              mid_ed = pd.DataFrame(mid_ed, columns=["x","y","z"],)
              mid_es = pd.DataFrame(mid_es, columns=["x","y","z"],)
              ED_all = pd.concat([ED_data, mid_ed],axis=0)
              ES_all = pd.concat([ES_data, mid_es],axis=0)
              # Step 4 - Transform ED_all from cartesian to cylindrical coordinates
              ED_datan = np.zeros((len(ED_all.iloc[:,0]),3))
              ES_datan = np.zeros((len(ES_all.iloc[:,0]),3))
              for iE in range (0,(len(ED_all.iloc[:,0]))):
                     ED_datan[iE,:] = array(cart2cylc(ED_all.iloc[iE,0],ED_all.iloc[iE,1],ED_all.iloc[iE,2]))
                     ES_datan[iE,:] = array(cart2cylc(ES_all.iloc[iE,0],ES_all.iloc[iE,1],ES_all.iloc[iE,2]))
                     continue
              ED_epi = ED_datan[0:(len(EDepi_data[:,0])),:]
              ED_endo = ED_datan[(len(EDepi_data[:,0])):(len(ED_data.iloc[:,0])),:]
              ES_epi = ES_datan[0:(len(ESepi_data[:,0])),:]
              ES_endo = ES_datan[(len(ESepi_data[:,0])):(len(ES_data.iloc[:,0])),:]
              mid_edn = ED_datan[(len(ED_data.iloc[:,0])):(len(ED_all.iloc[:,0])),:]
              mid_esn = ES_datan[(len(ES_data.iloc[:,0])):(len(ES_all.iloc[:,0])),:]
              # Step 5 - Compute 1% of all data points in LV and find the 1% knn in ED and ES surface get the knns only for the middle surface
              sc_1 = round(1*len(ED_all.iloc[:,0])/100)
              # find the sc_1 knn in middle surface for ES
              nPoints = np.arange(len(ED_epi[:,0]))
              nbrs3 = NearestNeighbors(n_neighbors=sc_1, algorithm='auto').fit(ES_datan[0:(len(ES_data.iloc[:,0])),:], mid_esn)
              distances_esm, con_es = nbrs3.kneighbors(mid_esn)
              attach_str = np.zeros((len(EDepi_data[:,0]),18))
              for iN in range(0,(len(EDepi_data[:,0]))):
                     diff_ed = abs(mid_edn[iN,:] - ED_datan[con_es[iN,:],:])
                     diff_es = abs(mid_esn[iN,:] - ES_datan[con_es[iN,:],:])
                     # Step 6 - create an mxlist with nPoint=nrow(mid_surf) and 1x6rows-1:sc_1columns for each point
                     mxlist = np.concatenate([diff_ed,diff_es], axis =1)
                     # Step 6 - Compute strain
                     attach_str[iN,:] = etens(mxlist)
                     continue
              # Get only the surface with the cartesian coordinates
              attach_ed = EDepi_data
              attach_es = ESepi_data
              #attach_str_new = np.zeros((len(EDepi_data[:,0]),len(attach_str[0,:])))
              #for iN in range(0,len(attach_str[0,:])):
              #kde = gaussian_kde(attach_str[:,0], bw_method=0.2)
              #attach_str_n = kde.evaluate(nPoints)
              #       continue
              neopheno = np.concatenate([ED_epi, ES_epi, attach_ed, attach_es, attach_str], axis=1)
              neopheno = pd.DataFrame(neopheno)
              neopheno.columns = ["EDx", "EDy", "EDz","ESx", "ESy", "ESz","EDcx", "EDcy", "EDcz","EScx", "EScy", "EScz",
                                  "ELRR","ELRT","ELRZ","ELTR","ELTT","ELTZ","ELZR","ELZT","ELZZ","EERR","EERT","EERZ","EETR",
                                  "EETT","EETZ","EEZR","EEZT","EEZZ"]
              neopheno.to_csv(os.path.join("middle_atlas/neopheno_"+str(ir)+".txt"), index=False, header=True)
              #plt.plot(nPoints, attach_str_n)
              #plt.savefig("images/Err.png") 
              print(ir)
              ir+=1
              continue
       continue


       

       """
       # plot the full ES mesh
       Data = ES_all
       import plotly.graph_objects as go
       ax = dict(title_text = "",
                 showgrid = False, 
                 zeroline = False, 
                 showline = False,
                 showticklabels = False,
                 showbackground= False)
       fig = go.Figure(data=[go.Scatter3d(
                     x=Data.x, y=Data.y, z=Data.z,
                     mode='markers',
                     marker=dict(
                                   size=0.6,cauto=False,
                                   color="#990000"))])
       fig.show()
       #if not os.path.exists("images"):
       #       os.mkdir("images")
       fig.write_html("images/plot_mid.html", auto_open=True)
       """

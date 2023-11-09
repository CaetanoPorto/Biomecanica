#!/usr/bin/env python
# coding: utf-8

# # Trabalho 2 Biomecanica
# ### Professor: Paulo Preto
# ### Aluno: Guilherme Caetano Porto
# ### Universidade de São Paulo

# In[1]:


# carregando as bibliotecas necessarias
import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
import pandas as pd
# from scipy.signal import find_peaks
import scipy as sp
import matplotlib.pyplot as plt
import sys
#from rich import print


# In[2]:


# Lendo os arquivos
cp2d_c1 = pd.read_csv('Dados/c1cal2rec3d.dat', delimiter=' ', header=None)
cp2d_c2 = pd.read_csv('Dados/c2cal2rec3d.dat', delimiter=' ', header=None)
cp2dc1 = np.asarray(cp2d_c1)
cp2dc2 = np.asarray(cp2d_c2)

cp3d = pd.read_csv('Dados/ref3d_v2.ref', delimiter=' ',header=None)
cp3d = np.asarray(cp3d)


# In[3]:


cp3d = np.asarray(cp3d)
cp2d = np.asarray(cp2dc1)

m = np.size(cp3d[:, 0], 0)
M = np.zeros([m * 2, 11])
N = np.zeros([m * 2, 1])

for i in range(m):
    M[i*2,:] = [cp3d[i,0], cp3d[i,1], cp3d[i,2] ,1, 0, 0, 0, 0, -cp2d[i, 0] * cp3d[i, 0], -cp2d[i, 0] * cp3d[i, 1], -cp2d[i, 0] * cp3d[i, 2]]
    M[i*2+1,:] = [0 , 0, 0, 0, cp3d[i,0], cp3d[i,1], cp3d[i,2],1, -cp2d[i,1] * cp3d[i,0],-cp2d[i,1] * cp3d[i,1], -cp2d[i,1] * cp3d[i,2]]
    N[[i*2,i*2+1],0] = cp2d[i, :]

Mt = M.T
M1 = inv(Mt.dot(M))
MN = Mt.dot(N)

DLT_c1 = (M1).dot(MN).T

print(DLT_c1)


# In[4]:


cp3d = np.asarray(cp3d)
cp2d = np.asarray(cp2dc2)

m = np.size(cp3d[:, 0], 0)
M = np.zeros([m * 2, 11])
N = np.zeros([m * 2, 1])

for i in range(m):
    M[i*2,:] = [cp3d[i,0], cp3d[i,1], cp3d[i,2] ,1, 0, 0, 0, 0, -cp2d[i, 0] * cp3d[i, 0], -cp2d[i, 0] * cp3d[i, 1], -cp2d[i, 0] * cp3d[i, 2]]
    M[i*2+1,:] = [0 , 0, 0, 0, cp3d[i,0], cp3d[i,1], cp3d[i,2],1, -cp2d[i,1] * cp3d[i,0],-cp2d[i,1] * cp3d[i,1], -cp2d[i,1] * cp3d[i,2]]
    N[[i*2,i*2+1],0] = cp2d[i, :]

Mt = M.T
M1 = inv(Mt.dot(M))
MN = Mt.dot(N)

DLT_c2 = (M1).dot(MN).T

print(DLT_c2)


# In[5]:


# Juntando o as cordenadas de DLT_c1 e DLT_c2

DLT_1_2 = np.append(DLT_c1 , DLT_c2 ,axis = 0)
cc3d = np.zeros([len(cp2dc1), 3])


# In[6]:


# %% Reconstruction 3D

def r3d(DLT_1_2, cc2ds):
    DLT_1_2 = np.asarray(DLT_1_2)
    cc2ds = np.asarray(cc2ds)
    
    m = len(DLT_1_2)
    M = np.zeros([2 * m, 3])
    N = np.zeros([2 * m, 1])

    for i in range(m):
        M[i*2,:] = [DLT_1_2[i,0]-DLT_1_2[i,8]*cc2ds[i,0], DLT_1_2[i,1]-DLT_1_2[i,9]*cc2ds[i,0], DLT_1_2[i,2]-DLT_1_2[i,10]*cc2ds[i,0]]
        M[i*2+1,:] = [DLT_1_2[i,4]-DLT_1_2[i,8]*cc2ds[i,1],DLT_1_2[i,5]-DLT_1_2[i,9]*cc2ds[i,1],DLT_1_2[i,6]-DLT_1_2[i,10]*cc2ds[i,1]]
        Np1 = cc2ds[i,:].T
        Np2 = [DLT_1_2[i,3],DLT_1_2[i,7]]
        N[[i*2,i*2+1],0] = Np1 - Np2

    cc3d = inv(M.T.dot(M)).dot((M.T.dot(N)))
    
    return cc3d


# In[7]:


# Chamando a função de reconstrução 3D

for i in range(len(cp2dc1)):
        cc2ds = np.append([cp2dc1[i, :]], [cp2dc2[i, :]], axis=0)
        cc3d[i, :] = r3d(DLT_1_2, cc2ds).T

print(cc3d)


# ### Carregando as cordenadas em um plano 3d

# In[8]:


# Salvando como arquivo txt

np.savetxt("cordfinal"+'.3d', cc3d, fmt='%.10f')

# Carregando os pontos reconstruídos 

cc3d = np.loadtxt("cordfinal.3d")

# Exibir os pontos 3D reconstruídos no plano 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cc3d[:, 0], cc3d[:, 1], cc3d[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Reconstrução 3D')
plt.show()


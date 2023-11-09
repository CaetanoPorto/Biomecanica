#!/usr/bin/env python
# coding: utf-8

# # Trabalho Biomecanica, Guilherme Porto
# ## N USP: 13688753

# In[1]:


import pandas as pd


# In[2]:


uri = 'Dados/cruzreta.xlsx'
dados = pd.read_excel(uri)
dados


# In[3]:


yp1 = dados.loc[0 , 'yp1' ] 
yc1 = dados.loc[0 , 'yc1' ] 
xp1 = dados.loc[0 , 'xp1' ] 
xc1 = dados.loc[0 , 'xc1' ] 

# Coeficiente angular primeira reta
mr1 = (yp1 - yc1 ) / (xp1 - xc1)
mr1


# In[4]:


yp2 = dados.loc[0 , 'yp2' ] 
yc2 = dados.loc[0 , 'yc2' ] 
xp2 = dados.loc[0 , 'xp2' ] 
xc2 = dados.loc[0 , 'xc2' ] 

# Coeficiente angular segunda reta
mr2 = (yp2 - yc2 ) / (xp2 - xc2)
mr2


# In[9]:


# Cordenadas de encontro das Retas

eixo_x = (-(mr2*xc2) + yc2 + (mr1*xc1) - yc1) / (mr1 - mr2)
eixo_y = (mr1*eixo_x) - (mr1*xc1) + yc1
print(eixo_x)
print(eixo_y)


# In[ ]:





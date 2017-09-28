# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:24:07 2017

@author: boscence
"""

#%%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('xxx')

for i in df.columns: print(i)

cols_to_use = ['FULLMODEL',
                'VEHICLECODE_PK',
               'WEIGHTMAX',
               'CYLINDERNB',
               'POWERCC',
               'CO2EMISSION',
               'CONSUMPTION',
               'TANKCAPACITY',
               'BODYDESCRIPTION',
               'PRICEWITHOUTOPTLISTWITHOUTVAT',
               'PRICEOPTLISTWITHOUTVAT',
               #'DOORNB'
               ]

df_cluster = df[cols_to_use]
df_cluster = df_cluster.dropna()
df_cluster = df_cluster.sample(n=5000)
#%%

#body_dummies = pd.get_dummies(df_cluster['BODYDESCRIPTION'])

#df_cluster.drop('BODYDESCRIPTION',1)

#data = pd.concat([df_cluster, body_dummies], axis=1).drop(['BODYDESCRIPTION','FULLMODEL'],1)
data = df_cluster.drop(['BODYDESCRIPTION','FULLMODEL'],1)

#%%

data.isnull().sum()

#%%

scaler = StandardScaler()
X_std = scaler.fit_transform(data)

#%%
dframe = X_std #data
#dframe = data #data

clt = KMeans(n_clusters=4, random_state=0)
model = clt.fit(dframe)


df_cluster['clusters'] = model.labels_

df_cluster.groupby('clusters').size()

pca = decomposition.PCA(n_components=2)
df_cluster['x'] = pca.fit_transform(dframe)[:,0]
df_cluster['y'] = pca.fit_transform(dframe)[:,1]
#%%
groups = df_cluster.groupby('clusters')

# Plot
fig, ax = plt.subplots(figsize=(8,8))

ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='.', linestyle='', ms=5, label=name)
ax.legend(loc=0)

plt.show();
#silhouette_score(X_std,model.labels_,metric='euclidean')

#%%
df_cluster.groupby('clusters').size()
df_cluster[['clusters','FULLMODEL']][df_cluster['clusters']==0]
df_cluster[['clusters','FULLMODEL']][df_cluster['clusters']==1]
df_cluster[['clusters','FULLMODEL']][df_cluster['clusters']==2]
df_cluster[['clusters','FULLMODEL']][df_cluster['clusters']==3]


#%%
score_plot= []
for n_clusters in range(2,7,1):
    cluster_model = KMeans(n_clusters=n_clusters)
    cluster_labels = cluster_model.fit_predict(X_std)
    silhouette_avg = silhouette_score(X_std,cluster_labels,metric='euclidean')
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is:", silhouette_avg)
    score_plot.append(silhouette_avg)

#%%

fig, ax = plt.subplots()
fig.canvas.draw()
labels = [str(i) for i in range(2,8,1)]
ax.set_xticklabels(labels)
plt.plot(score_plot)
plt.show()


#%%
if 'clusters' in cols_to_use:
    cols_to_use.remove('cl')
    
if 'FULLMODEL' in cols_to_use:
    cols_to_use.remove('FULLMODEL')    

if 'BODYDESCRIPTION' in cols_to_use:
    cols_to_use.remove('BODYDESCRIPTION')    


for i in df_cluster:
    if i in cols_to_use:
        print(i)
        fig, ax = plt.subplots(figsize=(14,6))
        ylimu = (df_cluster[i].max())+(df_cluster[i].max())*.05
        df_cluster[[i,'clusters']].boxplot(ax=ax,by='clusters',figsize=(14,6))
        ax.margins(y=0.05)
        plt.show();

#%reset

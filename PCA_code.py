#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:48:19 2023

@author: jamescarney
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:26:27 2021

@author: jamescarney
"""


import pandas as pd



import plotly.express as px


import re

import seaborn as sns
sns.set()


from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


import matplotlib.pyplot as plt



vecs_ = pd.read_csv('/Users/jamescarney/Desktop/Quantifying_interdisciplinarity/TF-IDF_data.csv', index_col = 0)

disciplines = vecs_.index

pca_1 = PCA(n_components = 3)

comps_1 = pca_1.fit_transform(vecs_)


pc_df_1 = pd.DataFrame(data = comps_1, columns = ['PC'+str(i) for i in range(1, comps_1.shape[1]+1)])

clustering = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward').fit(vecs_)


#vecs_ = vecs_.reset_index()

vecs_df = pc_df_1

vecs_df['discipline'] = disciplines

vecs_df['cluster label'] = [str(i) for i in clustering.labels_]

"""

vecs_df = pd.concat([vecs_, pc_df_1], axis = 1)




vecs_df['cluster label'] = clustering.labels_
"""

fig = px.scatter_3d(vecs_df, x='PC1', y='PC2', z='PC3', hover_data = ['discipline'], color = 'cluster label')
fig.update_traces(marker=dict(size=10,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig.write_html('/Users/jamescarney/Desktop/Quantifying_interdisciplinarity/ID_3d.html')

# =============================== SETTING ====================================
import sys
import pandas as pd
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding,TSNE, SpectralEmbedding
from sklearn.decomposition import PCA as sklearnPCA

argv = sys.argv
input_data = argv[1]
outdir = argv[2]
k = int(argv[3])
n_pc = int(argv[4])
# k=10
# n_pc = 40
# =============================== Data loading ====================================
#data = pd.read_csv(input_data, index_col=0,delimiter="\t")
data = pd.read_pickle(input_data)

# =============================== PCA ====================================
sklearn_pca = sklearnPCA(n_components=100,svd_solver='arpack',random_state=42)
trans = sklearn_pca.fit(np.transpose(data.values))
X_pca = trans.transform(np.transpose(data.values))

# =============================== Embedding ====================================
np.random.seed(2)
reducer = SpectralEmbedding(n_neighbors=k, 
                                 n_components=2,
                                 n_jobs = 4,
                                 eigen_solver = None,
                            random_state=10)
trans = reducer.fit(X_pca[:,0:n_pc])
embed = trans.embedding_

# =============================== Export ====================================
out = pd.DataFrame({"SAMPLE.ID" : np.array(data.columns),
                   "Component1" : embed[:,0],
                   "Component2" : embed[:,1]})
out.to_csv(outdir+'/k'+str(k)+'_pc'+str(n_pc)+'.gzip', sep="\t",index=False)  


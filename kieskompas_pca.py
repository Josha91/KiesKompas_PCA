import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
from sklearn.decomposition import PCA
import pandas as pd
import IPython
import glob

def plot_logo(party,pos,ax=None):
	"""
	party -- partyname
	pos   -- (x,y) PCA coordinates
	"""
	if ax is None: 
		ax = plt.gca()

	im_file = glob.glob('logos/*%s*'%party)
	im = OffsetImage(plt.imread(im_file[0]),zoom=0.2)
	ab = AnnotationBbox(im,pos,xycoords='data',frameon=False)
	ax.add_artist(ab)

data = pd.read_csv('kieskompas.csv')

pca = PCA(n_components=2)
pca.fit(data)

print(pca.explained_variance_ratio_)

fig,ax = plt.subplots(1)
for i in np.arange(len(data.columns)):
	pos = (pca.components_[0,i],pca.components_[1,i])
	plot_logo(data.columns[i],pos,ax=ax)
ax.update_datalim(np.column_stack([pca.components_[0,:],pca.components_[1,:]]))
ax.autoscale()
x_mean = np.median(pca.components_[0,:])
y_mean = np.median(pca.components_[1,:])
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
ax.plot(ax.get_xlim(),[y_mean]*2,linestyle='dashed',color='k')
ax.plot([x_mean]*2,ax.get_ylim(),linestyle='dashed',color='k')
ax.set_xlabel('Principal component 1',fontsize=16)
ax.set_ylabel('Principal component 2',fontsize=16)
plt.show()
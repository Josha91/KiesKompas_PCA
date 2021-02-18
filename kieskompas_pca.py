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
	im = OffsetImage(plt.imread(im_file[0]),zoom=0.2,zorder=1)
	ab = AnnotationBbox(im,pos,xycoords='data',frameon=False)
	ax.add_artist(ab)

def morph_kieskompas_to_pca(df,ax=None):
	"""
	df is a dataframe with: 
	(1) names of political parties
	(2) coordinates in kieskompas
	(3) coordinates in PCA
	"""
	pass
	#if ax is None: fig, ax = plt.subplots(1)

class Animation_kieskompas:
	def __init__(self):
		data = pd.read_csv('kieskompas.csv')
		pca = PCA(n_components=2)
		self.pca = pd.DataFrame(pca.fit(data).components_.T,index=data.keys(),\
						columns=['leftright','consprog'])
		self.kk = pd.read_csv('coords_kieskompas.csv',index_col='party')

		self.standardize()

		self.data = pd.merge(left=self.kk,right=self.pca,left_on=self.kk.index,\
							right_on=self.pca.index).set_index('key_0')

	def standardize(self):
		def rescale(x):
			return (x-np.mean(x))*2/(x.max()-x.min()) 
		self.pca['leftright'] = rescale(self.pca['leftright'])
		self.pca['consprog'] = rescale(self.pca['consprog'])

		self.kk['leftright'] = rescale(self.kk['leftright'])
		self.kk['consprog'] = rescale(self.kk['consprog'])

	def plot(self):
		N = 30
		fig,ax = plt.subplots(1)
		for i in np.arange(N):
			ax.cla()
			ax.set_xlim([-1.5,1.5])
			ax.set_ylim([-1.5,1.5])
			ax.plot([-1.5,1.5],[0,0],color='k',zorder=0)
			ax.plot([0,0],[-1.5,1.5],color='k',zorder=0)
			ax.set_aspect('equal')
			plt.axis('off')
			#plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
			for j in np.arange(self.data.shape[0]):
				pos_x = self.data['leftright_x'].iloc[j]*(1-i/N)+(i/N)*\
								self.data['leftright_y'].iloc[j]
				pos_y = self.data['consprog_x'].iloc[j]*(1-i/N)+(i/N)*\
								self.data['consprog_y'].iloc[j]
				plot_logo(self.data.index[j],(pos_x,pos_y),ax=ax)
				ax.grid()
			if i == 0:
				ax.text(0,-1.7,'Conservatief',ha='center',fontweight='bold')
				ax.text(0,1.7,'Progressief',ha='center',fontweight='bold')
				ax.text(-1.7,0,'Links',rotation=90,va='center',fontweight='bold')
				ax.text(1.7,0,'Rechts',rotation=-90,va='center',fontweight='bold')
				#ax.set_title('Kieskompas',fontsize=18,fontweight='bold')
				plt.savefig('kieskompas.png',dpi=300)
			#plt.pause(0.1)
		ax.text(0,-1.7,'Principal component 1',fontsize=14,ha='center')
		ax.text(-1.7,0,'Principal component 2',fontsize=14,rotation=90,va='center')
		ax.set_title("PCA plane",fontsize=18,fontweight='bold')
		#ax.set_ylabel('Principal component 2',fontsize=16)
		plt.savefig('pca_kieskompas.png',dpi=300)
		plt.show()

		
def old():
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
	ax.plot(ax.get_xlim(),[y_mean]*2,linestyle='dashed',color='k',zorder=0)
	ax.plot([x_mean]*2,ax.get_ylim(),linestyle='dashed',color='k',zorder=0)
	ax.set_xlabel('Principal component 1',fontsize=16)
	ax.set_ylabel('Principal component 2',fontsize=16)
	#plt.savefig('kieskompas_PCA.png',dpi=300)
	plt.show()

	IPython.embed()
	coord = pd.read_csv('coords_kieskompas.csv',index_col='party')
	fig,ax = plt.subplots(1)
	for i in np.arange(coords['leftright'].size):
		plot_logo(coords.index[i],coord.loc[coords.index[i]],ax=ax)
	ax.set_xlim([-400,400])
	ax.set_ylim([-400,400])
	ax.plot(ax.get_xlim(),[0]*2,linestyle='dashed',color='k',zorder=0)
	ax.plot([0]*2,ax.get_ylim(),linestyle='dashed',color='k',zorder=0)
	plt.axis('off')
	ax.set_aspect('equal')
	ax.text(0,-430,'Conservatief',ha='center')
	ax.text(0,430,'Progressief',ha='center')
	ax.text(-450,0,'Links',rotation=90,va='center')
	ax.text(430,0,'Rechts',rotation=-90,va='center')
	plt.show()

a = Animation_kieskompas()
a.plot()
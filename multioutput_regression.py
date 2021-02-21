import numpy as np 
import pandas as pd 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt 
import IPython
import seaborn as sns
from kieskompas_pca import plot_logo

suffix = '_2017'

data = pd.read_csv('kieskompas%s.csv'%suffix)
kieskompas = pd.read_csv('coords_kieskompas%s.csv'%suffix,index_col='party')

clf = MultiOutputRegressor(Ridge(random_state=123)).fit(data.T,kieskompas)

realizations = []
N =100000
for _ in np.arange(N):
	tmp = np.random.choice(np.arange(-2,3),30)
	realizations.append(clf.predict(tmp))
realizations = np.asarray(realizations).squeeze()

fig,ax = plt.subplots(1)
ax.scatter(kieskompas['leftright'],kieskompas['consprog'],edgecolor='k',zorder=1)
ax.scatter(realizations[:,0],realizations[:,1],s=1,zorder=0)
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
ax.plot([0,0],ax.get_ylim(),color='k')
ax.plot(ax.get_xlim(),[0,0],color='k')
q1 = np.sum((realizations[:,0]>0)&(realizations[:,1]>0))
q2 = np.sum((realizations[:,0]<0)&(realizations[:,1]>0))
q3 = np.sum((realizations[:,0]>0)&(realizations[:,1]<0))
q4 = np.sum((realizations[:,0]<0)&(realizations[:,1]<0))
ax.text(400,400,'%.1f'%(100*q1/N))
ax.text(-600,400,'%.2f'%(100*q2/N))
ax.text(400,-400,'%.2f'%(100*q3/N))
ax.text(-600,-400,'%.2f'%(100*q4/N))
plt.show()

df = pd.DataFrame(realizations,columns=['leftright','consprog'])

#fig,ax = plt.subplots(1)
p = sns.jointplot(x='leftright',y='consprog',data=df,s=10,alpha=0.5,color='grey')
for party in kieskompas.index:
	plot_logo(party,kieskompas.T[party],ax=p.ax_joint)
p.ax_joint.legend_.remove()
p.ax_joint.set_xlim(ax.get_xlim())
p.ax_joint.set_ylim(ax.get_ylim())
p.ax_joint.plot([0,0],p.ax_joint.get_ylim(),color='k')
p.ax_joint.plot(p.ax_joint.get_xlim(),[0,0],color='k')
p.ax_marg_x.set_ylim(p.ax_marg_x.get_ylim())
p.ax_marg_x.plot([0,0],p.ax_marg_x.get_ylim(),color='k')
p.ax_marg_y.set_xlim(p.ax_marg_y.get_xlim())
p.ax_marg_y.plot(p.ax_marg_y.get_xlim(),[0,0],color='k')
p.ax_joint.set_xticklabels([])
p.ax_joint.set_yticklabels([])
p.ax_joint.set_xlabel('')
p.ax_joint.set_ylabel('')
p.ax_joint.text(p.ax_joint.get_xlim()[0]+50,p.ax_joint.get_ylim()[0]-50,'Links')
p.ax_joint.text(p.ax_joint.get_xlim()[1]-250,p.ax_joint.get_ylim()[0]-50,'Rechts')
p.ax_joint.text(p.ax_joint.get_xlim()[0]-50,p.ax_joint.get_ylim()[0]+300,'Conservatief',rotation=90)
p.ax_joint.text(p.ax_joint.get_xlim()[0]-50,p.ax_joint.get_ylim()[1]-100,'Progressief',rotation=90)
plt.savefig('random_realisations_kieskompas%s.png'%suffix,dpi=300)

IPython.embed()
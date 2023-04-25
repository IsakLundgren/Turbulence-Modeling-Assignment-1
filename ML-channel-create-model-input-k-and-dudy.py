import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from joblib import dump, load
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker

plt.rcParams.update({'font.size': 22})

viscos=1/5200

plt.close('all')
plt.interactive(True)

# load DNS data
DNS_mean=np.genfromtxt("LM_Channel_5200_mean_prof.dat",comments="%")
y_DNS=DNS_mean[:,0]
yplus_DNS=DNS_mean[:,1]
u_DNS=DNS_mean[:,2]
dudy_DNS=np.gradient(u_DNS,y_DNS)

DNS_stress=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat",comments="%")
uu_DNS=DNS_stress[:,2]
vv_DNS=DNS_stress[:,3]
ww_DNS=DNS_stress[:,4]
uv_DNS=DNS_stress[:,5]
k_DNS=0.5*(uu_DNS+vv_DNS+ww_DNS)

DNS_RSTE=np.genfromtxt("LM_Channel_5200_RSTE_k_prof.dat",comments="%")
eps_DNS=DNS_RSTE[:,7]/viscos # it is scaled with ustar**4/viscos
# fix wall
eps_DNS[0]=eps_DNS[1]
vist_DNS=abs(uv_DNS)/dudy_DNS

uv_DNS=abs(uv_DNS)

# vist = cmu*k**2/eps
# omega = eps/k = eps/(vist*eps/cmu)**0.5 = (eps/vist/cmu)**0.5
omega_DNS=(eps_DNS/0.09/vist_DNS)**0.5
cmu_DNS=uv_DNS/(k_DNS*dudy_DNS)*omega_DNS

uv_DNS_org=uv_DNS
yplus_DNS_org=yplus_DNS
dudy_DNS_org=dudy_DNS
k_DNS_org=k_DNS
# Input data: dudy
# choose` DNS data for yplus > 100 and < 2000
index=np.nonzero((yplus_DNS > 100 )  & (yplus_DNS< 2000 ))
index_org=index
yplus_DNS=yplus_DNS[index]
dudy_DNS=dudy_DNS[index]
omega_DNS=omega_DNS[index]
uv_DNS=uv_DNS[index]
cmu_DNS=cmu_DNS[index]
k_DNS=k_DNS[index]

# find min & max
dudy_min = np.min(dudy_DNS)
dudy_max = np.max(dudy_DNS)

k_min = np.min(k_DNS)
k_max = np.max(k_DNS)


cmu_all_data=cmu_DNS
yplus_DNS_all_data=yplus_DNS
# input dudy
dudy_all_data=dudy_DNS
# input dudy
k_all_data=k_DNS

# create new indices for all data (which goes from 0 to len(uv_all_data)
index= np.arange(0,len(cmu_all_data), dtype=int)

# number of elements of test data, 20%
n_test=int(0.2*len(cmu_all_data))

# the rest is for training data
n_svr=len(cmu_all_data)-n_test

# pick 20% elements randomly (test data)
index_test=np.random.choice(index, size=n_test, replace=False)
# pick every 5th elements 
#index_test=index[::5]
dudy_test=dudy_all_data[index_test]
k_test=k_all_data[index_test]
yplus_DNS_test=yplus_DNS_all_data[index_test]
cmu_test=cmu_all_data[index_test]
n_test=len(dudy_test)
print('n_test',n_test)

# delete testing data from 'all data' => training data
dudy_in=np.delete(dudy_all_data,index_test)
k_in=np.delete(k_all_data,index_test)
cmu_out=np.delete(cmu_all_data,index_test)
n_svr=len(cmu_out)

# re-shape
dudy_in=dudy_in.reshape(-1, 1)
k_in=k_in.reshape(-1, 1)

# scale input data 
scaler_dudy=StandardScaler()
scaler_k=StandardScaler()
dudy_in=scaler_dudy.fit_transform(dudy_in)
k_in=scaler_k.fit_transform(k_in)

# setup X (input) and y (output)
X=np.zeros((n_svr,2))
y=cmu_out
X[:,0]=dudy_in[:,0]
X[:,1]=k_in[:,0]

print('starting SVR')

# choose Machine Learning model
C=10
#C=1000
eps=0.001
model = SVR(kernel='rbf', epsilon = eps, C = C)

# Fit the model
svr = model.fit(X, y.flatten())

#  re-shape test data
dudy_test=dudy_test.reshape(-1, 1)
k_test=k_test.reshape(-1, 1)

# scale test data
dudy_test=scaler_dudy.transform(dudy_test)
k_test=scaler_k.transform(k_test)

# setup X (input) for testing (predicting)
X_test=np.zeros((n_test,2))
X_test[:,0]=dudy_test[:,0]
X_test[:,1]=k_test[:,0]

# predict uv
cmu_predict= model.predict(X_test)

# find difference between ML prediction and target
cmu_error=np.std(cmu_predict-cmu_test)/\
(np.mean(cmu_predict**2))**0.5
print('\nRMS error using ML turbulence model',cmu_error)

########################################## uv 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(yplus_DNS_org,-uv_DNS_org,'b-',label='all DNS data')
plt.plot(yplus_DNS[::10],-uv_DNS[::10],'ro',label='used DNS data')
plt.ylabel(r"$\overline{u'v'}$")
plt.xlabel("$y^+$")
plt.legend(loc="best",fontsize=18)
plt.axis([0, 5200, -1, 0])
plt.savefig('uv_DNS.png',bbox_inches='tight')


# Set Increments between points in a meshgrid
mesh_size = 0.05

# Find the parameter space
# Identify min and max values for input variables
x_min, x_max = X[:,0].min(), X[:,0].max()
y_min, y_max = X[:,1].min(), X[:,1].max()

# Return evenly spaced values based on a range between min and max
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)

# Create a meshgrid
xx, yy= np.meshgrid(xrange, yrange)

# Use model to create a prediction plane --- SVR
cmu_predict_xx_yy = model.predict(np.c_[xx.ravel(), yy.ravel()])
cmu_predict_xx_yy = cmu_predict_xx_yy.reshape(xx.shape)

# transform back to physical values (non-scaled)
xx_no_scale=scaler_dudy.inverse_transform(xx)
yy_no_scale=scaler_k.inverse_transform(yy)

####################################### cmu  2D scatter
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax=plt.gca()

# transform back to physical values (non-scaled)
dudy_test_no_scale=scaler_dudy.inverse_transform(dudy_test)
k_test_no_scale=scaler_k.inverse_transform(k_test)

# plot color surface
ax_plot=plt.pcolormesh(xx_no_scale,yy_no_scale, cmu_predict_xx_yy, vmin=0.6,vmax=1,cmap=plt.get_cmap('hot'),shading='gouraud')

# scatter plot of predicted cmu
plt.scatter(dudy_test_no_scale.flatten(),k_test_no_scale.flatten(),marker='o',s=10,c='black')

plt.axis([0,125,2,5])

#label axes
plt.xlabel(r'$\frac{\partial \bar{U}}{\partial y}$')
plt.ylabel(r'$k$')

# put horizontal colorbar at the top 
cbaxes = fig1.add_axes([0.35, 0.93, 0.33, 0.02])  # x_start, y_start, x_width, y_width [0--1]
clb=plt.colorbar(ax_plot,cax=cbaxes,orientation='horizontal')
clb.ax.tick_params(labelsize=11)
clb.ax.set_title(r'$C_\mu$',fontsize=11)

plt.savefig('2D-scatter-dudy-and-k-vs-uv-svr-and-test.png',bbox_inches='tight')

########################################## Cmu 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.2,bottom=0.20)
plt.plot(yplus_DNS_test,cmu_predict,'bo',label='svr')
plt.plot(yplus_DNS_test,cmu_test,'r+',label='DNS')
plt.legend(loc="best",fontsize=18)
plt.ylabel(r"$C_\mu$")
plt.xlabel("$y^+$")
plt.axis([100, 2000, 0.6, 1.1])
plt.savefig('cmu.png',bbox_inches='tight')



import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# load data from k-omega RANS
data = np.loadtxt('y_u_k_om_uv_5200-RANS-code.dat')
y_rans = data[:,0]
k_rans = data[:,2]
# interpolate to DNS grid
k_rans_DNS=np.interp(y_DNS, y_rans, k_rans)


# vist and diss of k-omega model agree well with DNS, but not k. Hence omega is taken from diss and vist
# vist = cmu*k**2/eps
# omega = eps/k = eps/(vist*eps/cmu)**0.5 = (eps/vist/cmu)**0.5
omega_DNS=(eps_DNS/0.09/vist_DNS)**0.5


# turbulence model: uv = -cmu*k/omega*dudy => cmu=-uv/(k*dudy)*omega
# Input data: abs(dudy), L_m
dudy_all_data = np.abs(dudy_DNS)
L_m_all_data = y_DNS

# output, to be predicted: f_m
f_m_all_data = np.divide(vist_DNS,np.multiply(L_m_all_data**2, dudy_all_data))

#TODO here the higher values of the stress are chosen for the turbulent boundary layer
# choose values for 30 < y+ < 1000
index_choose=np.nonzero((yplus_DNS > 30 )  & (yplus_DNS< 1000 ))
yplus_DNS=yplus_DNS[index_choose]
dudy_all_data = dudy_all_data[index_choose]
L_m_all_data = L_m_all_data[index_choose]
f_m_all_data = f_m_all_data[index_choose]

# create indices for all data
index= np.arange(0,len(f_m_all_data), dtype=int)

# number of elements of test data, 20%
n_test=int(0.2*len(f_m_all_data))

# pick 20% elements randomly (test data)
index_test=np.random.choice(index, size=n_test, replace=False)
# pick every 5th elements 
#index_test=index[::5]

dudy_test=dudy_all_data[index_test]
L_m_test = L_m_all_data[index_test]
f_m_out_test=f_m_all_data[index_test]
n_test=len(dudy_test)

# delete testing data from 'all data' => training data
dudy_in=np.delete(dudy_all_data,index_test)
L_m_in = np.delete(L_m_all_data,index_test)
f_m_out=np.delete(f_m_all_data,index_test)
n_svr=len(f_m_out)

# re-shape
dudy_in=dudy_in.reshape(-1, 1)
L_m_in = L_m_in.reshape(-1, 1)

# scale input data 
scaler_dudy=StandardScaler()
scaler_L_m=StandardScaler()
dudy_in=scaler_dudy.fit_transform(dudy_in)
L_m_in=scaler_L_m.fit_transform(L_m_in)

# setup X (input) and y (output)
X=np.zeros((n_svr,2))
y=f_m_out
X[:,0]=dudy_in[:,0]
X[:,1]=L_m_in[:,0]

print('starting SVR')

# choose Machine Learning model
#TODO change C value
C=1e+3
eps=1e-4
# use Linear model
#model = LinearSVR(epsilon = eps , C = C, max_iter=1000)
model = SVR(kernel='rbf', epsilon = eps, C = C)

# Fit the model
svr = model.fit(X, y.flatten())

#  re-shape test data
dudy_test=dudy_test.reshape(-1, 1)
L_m_test = L_m_test.reshape(-1, 1)

# scale test data
dudy_test=scaler_dudy.transform(dudy_test)
L_m_test=scaler_dudy.transform(L_m_test)

# setup X (input) for testing (predicting)
X_test=np.zeros((n_test,2))
X_test[:,0]=dudy_test[:,0]
X_test[:,1]=L_m_test[:,0]

# predict cmu
f_m_predict= model.predict(X_test)

# find difference between ML prediction and target
f_m_error=np.std(f_m_predict-f_m_out_test)/\
(np.mean(f_m_predict**2))**0.5
print('\nRMS error using ML turbulence model',f_m_error)

# plot predicted vs true values
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(f_m_out_test, f_m_predict)
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')
ax.set_title('SVR Model Performance')
plt.show(block=True)


plt.savefig('scatter-cmu-vs-dudy-svr-and-test.png',bbox_inches='tight')


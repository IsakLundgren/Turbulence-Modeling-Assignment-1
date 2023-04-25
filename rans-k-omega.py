import scipy.io as sio
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#from tdma import tdma
#from IPython import display
plt.rcParams.update({'font.size': 22})

plt.interactive(True)

plt.close('all')

# This file can be downloaded at

#

# exemple of 1d Channel flow with a k-omegaa model. Re=u_tau*h/nu=5200 (h=half 
# channel height).
#
# Discretization described in detail in
# http://www.tfd.chalmers.se/~lada/comp_fluid_dynamics/

# max number of iterations
niter=25000

plt.rcParams.update({'font.size': 22})


#TODO inserting the machine learned cmu
from joblib import load

VistModel = True

folder = "./"

if VistModel:
      filename = str(folder) + "model-svr.bin"
      model = load(str(folder)+"nut-model-svr.bin")
      scaler_dudy = load(str(folder)+"nut-scalar-dudy-svr.bin")
      scaler_vist = load(str(folder)+"nut-scalar-vist-svr.bin")
      dudy_min, dudy_max, vist_min, vist_max = np.loadtxt(str(folder)+"vist-svr.txt")
else:
      filename = str(folder) + "model-svr.bin"
      model = load(str(folder)+"model-svr.bin")
      scaler_dudy = load(str(folder)+"scalar-dudy-svr.bin")
      dudy_min, dudy_max = np.loadtxt(str(folder)+"dudy-svr.txt")

SVR = True


# friction velocity u_*=1
# half channel width=1
#

# create the grid

#nj=30 # coarse grid
nj=98 # fine grid
njm1=nj-1
#yfac=1.6 # coarse grid
yfac=1.15 # fine grid
dy=0.1
yc=np.zeros(nj)
delta_y=np.zeros(nj)
yc[0]=0.
for j in range(1,int(nj/2)):
    yc[j]=yc[j-1]+dy
    dy=yfac*dy

ymax=yc[int(nj/2)-1]

# cell faces
for j in range(0,int(nj/2)):
   yc[j]=yc[j]/ymax
   yc[nj-1-j]=2.-yc[j-1]
yc[nj-1]=2.

# cell centres
yp=np.zeros(nj)
for j in range(1,nj-1):
   yp[j]=0.5*(yc[j]+yc[j-1])
yp[nj-1]=yc[nj-1]

# viscosity
viscos=1./5200.

# under-relaxation
urf=0.5

# plot k for each iteration at node jmon
jmon=8 

# turbulent constants 
c_omega_1= 5./9.
c_omega_2=3./40.
prand_omega=2.0
prand_k=2.0
cmu=0.09 

small=1.e-10
great=1.e10

# initialaze
u=np.zeros(nj)
k=np.ones(nj)*1.e-4
y=np.zeros(nj)
om=np.ones(nj)*1.
vist=np.ones(nj)*100.*viscos
dn=np.zeros(nj)
ds=np.zeros(nj)
dy_s=np.zeros(nj)
fy=np.zeros(nj)
C_mu=np.ones(nj)
tau_w=np.zeros(niter)
k_iter=np.zeros(niter)
om_iter=np.zeros(niter)


# do a loop over all nodes (except the boundary nodes)
for j in range(1,nj-1):

# compute dy_s
   dy_s[j]=yp[j]-yp[j-1]

# compute dy_n
   dy_n=yp[j+1]-yp[j]

# compute deltay
   delta_y[j]=yc[j]-yc[j-1]
 
   dn[j]=1./dy_n
   ds[j]=1./dy_s[j]

# interpolation factor
   del1=yc[j]-yp[j]
   del2=yp[j+1]-yc[j]
   fy[j]=del1/(del1+del2)

u[1]=0.


vist[0]=0.
vist[nj-1]=0.
k[0]=0.
k[nj-1]=0.


su=np.zeros(nj)
sp=np.zeros(nj)
an=np.zeros(nj)
as1=np.zeros(nj)
ap=np.zeros(nj)
# do max. niter iterations
for n in range(1,niter):

    for j in range(1,nj-1):

# compute turbulent viscosity
      vist_old=vist[j]
      vist[j]=urf*C_mu[j]*k[j]/om[j]+(1.-urf)*vist_old

# solve u
    for j in range(1,nj-1):

# driving pressure gradient
      su[j]=delta_y[j]

      sp[j]=0.

# interpolate turbulent viscosity to faces
      vist_n=fy[j]*vist[j+1]+(1.-fy[j])*vist[j]
      vist_s=fy[j-1]*vist[j]+(1.-fy[j-1])*vist[j-1]

# compute an & as
      an[j]=(vist_n+viscos)*dn[j]
      as1[j]=(vist_s+viscos)*ds[j]

# boundary conditions for u
    u[0]=0.
    u[nj-1]=0.


    for j in range(1,nj-1):
# compute ap
      ap[j]=an[j]+as1[j]-sp[j]

# under-relaxate
      ap[j]= ap[j]/urf
      su[j]= su[j]+(1.0-urf)*ap[j]*u[j]

# use Gauss-Seidel
      u[j]=(an[j]*u[j+1]+as1[j]*u[j-1]+su[j])/ap[j]


# monitor the development of u_tau in node jmon
    tau_w[n]=viscos*u[1]/yp[1]

# print iteration info
    tau_target=1
    #print(f"\n{'---iter: '}{n:2d}, {'wall shear stress: '}{tau_w[n]:.2e}\n")
    print(f"\n{'---iter: '}{n:2d}, {'wall shear stress: '}{tau_w[n]:.2e},{'  tau_w_target='}{tau_target:.2e}\n")

# check for convergence (when converged, the wall shear stress must be one)
    ntot=n
    if abs(tau_w[n]-1) < 0.001:
# do at least 1000 iter 
        if n > 1000:
           print('Converged!')
           break

# solve k
    dudy=np.gradient(u,yp)
    dudy2=dudy**2
# Calculate new C_mu WARNING decoupled
    if(SVR):
      if(VistModel):
            dudy_clamped = np.zeros((nj,1))
            maximum = dudy_max * np.ones(nj)
            minimum = dudy_min * np.ones(nj)
            #Since model is only trained on positive test data, take absolute value of dudy. 
            #Since C_mu positive both in positive and negative dudy, this should make sense
            dudy_clamped[:,0] = np.maximum(np.minimum(np.abs(dudy),maximum),minimum) 
            vist_clamped = np.zeros((nj,1))
            maximum = vist_max * np.ones(nj)
            minimum = vist_min * np.ones(nj)
            vist_clamped[:,0] = np.maximum(np.minimum(vist,maximum),minimum)
            model_input = np.zeros((nj,2))
            dudy_transformed = scaler_dudy.transform(dudy_clamped)
            vist_transformed = scaler_dudy.transform(vist_clamped)
            model_input[:,0] = dudy_transformed[:,0]
            model_input[:,1] = vist_transformed[:,0]
            absuv = model.predict(model_input)
            C_mu = np.divide(absuv,np.multiply(dudy_clamped[:,0],vist_clamped[:,0]))
      else:
            dudy_clamped = np.zeros((nj,1))
            maximum = dudy_max * np.ones(nj)
            minimum = dudy_min * np.ones(nj)
            dudy_clamped[:,0] = np.maximum(np.minimum(dudy,maximum),minimum)
            C_mu = model.predict(scaler_dudy.transform(dudy_clamped))
         
    for j in range(1,nj-1):


# production term
      su[j]=vist[j]*dudy2[j]*delta_y[j]

# dissipation term
      ret=k[j]/(viscos*om[j])
      ret=max(ret,1.e-5)

      sp[j]=-cmu*om[j]*delta_y[j]

# compute an & as
      vist_n=fy[j]*vist[j+1]+(1.-fy[j])*vist[j]
      an[j]=(vist_n/prand_k+viscos)*dn[j]
      vist_s=fy[j-1]*vist[j]+(1.-fy[j-1])*vist[j-1]
      as1[j]=(vist_s/prand_k+viscos)*ds[j]

# boundary conditions for k
    k[0]=0.
    k[nj-1]=0.

    for j in range(1,nj-1):
# compute ap
      ap[j]=an[j]+as1[j]-sp[j]

# under-relaxate
      ap[j]= ap[j]/urf
      su[j]= su[j]+(1.-urf)*ap[j]*k[j]

# use Gauss-Seidel
      k[j]=(an[j]*k[j+1]+as1[j]*k[j-1]+su[j])/ap[j]


# monitor the development of k in node jmon
    k_iter[n]=k[jmon]

#****** solve om-eq.
    for j in range(1,nj-1):
# compute an & as
      vist_n=fy[j]*vist[j+1]+(1.-fy[j])*vist[j]
      an[j]=(vist_n/prand_omega+viscos)*dn[j]
      vist_s=fy[j-1]*vist[j]+(1.-fy[j-1])*vist[j-1]
      as1[j]=(vist_s/prand_omega+viscos)*ds[j]

# production term
      su[j]=c_omega_1*dudy2[j]*delta_y[j]

# dissipation term
      sp[j]=-c_omega_2*om[j]*delta_y[j]

# b.c. south wall
    dy=yp[1]
    omega=6.*viscos/0.075/dy**2
    sp[1]=-great
    su[1]=great*omega

# b.c. north wall
    dy=yp[nj-1]-yp[nj-2]
    omega=6.*viscos/0.075/dy**2
    sp[nj-2]=-great
    su[nj-2]=great*omega

    for j in range(1,nj-1):
# compute ap
      ap[j]=an[j]+as1[j]-sp[j]

# under-relaxate
      ap[j]= ap[j]/urf
      su[j]= su[j]+(1.-urf)*ap[j]*om[j]

# use Gauss-Seidel
      om[j]=(an[j]*om[j+1]+as1[j]*om[j-1]+su[j])/ap[j]

    om_iter[n]=om[jmon]

# compute shear stress
uv=-vist*dudy

DNS_mean=np.genfromtxt("LM_Channel_5200_mean_prof.dat",comments="%")
y_DNS=DNS_mean[:,0]
yplus_DNS=DNS_mean[:,1]
u_DNS=DNS_mean[:,2]

DNS_stress=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat",comments="%")
u2_DNS=DNS_stress[:,2]
v2_DNS=DNS_stress[:,3]
w2_DNS=DNS_stress[:,4]
uv_DNS=DNS_stress[:,5]


k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)



# plot u
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(u,yp,'b-',label="CFD")
plt.plot(u_DNS,y_DNS,'r-',label="DNS")
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")
plt.legend(loc="best",prop=dict(size=18))
plt.savefig('u_5200.png')

# plot u log-scale
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ustar=tau_w[ntot]**0.5
yplus=yp*ustar/viscos
uplus=u/ustar
plt.semilogx(yplus,uplus,'b-',label="CFD")
plt.semilogx(yplus_DNS,u_DNS,'r-',label="DNS")
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")
plt.axis([1, 5200, 0, 28])
plt.legend(loc="best",prop=dict(size=18))
plt.savefig('u_log-5200.png')

# plot k
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(k,yp,'b-',label="CFD")
plt.plot(k_DNS,y_DNS,'r-',label="DNS")
plt.legend(loc="best",prop=dict(size=18))
plt.xlabel('k')
plt.ylabel('y')
plt.savefig('k_5200.png')

# plot tau_w versus iteration number
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(tau_w[0:ntot],'b-')
#plt.plot(tau_w[0:ntot],'b-')
plt.title('wall shear stress')
plt.xlabel('Iteration number')
plt.ylabel('tauw')
plt.savefig('tauw.png')

# plot k(jmon) versus iteration number
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(k_iter[0:ntot],'b-')
#plt.plot(k_iter[0:ntot],'b-')
plt.title('k in node jmon')
plt.xlabel('Iteration number')
plt.ylabel('k')
plt.savefig('k_iter.png')

# plot om(jmon) versus iteration number
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(om_iter[0:ntot],'b-')
#plt.plot(om_iter[0:ntot],'b-')
plt.title('omega in node jmon')
plt.xlabel('Iteration number')
plt.ylabel('omega')
plt.savefig('om_iter.png')

# save data
data=np.zeros((nj,8))
data[:,0]=yp
data[:,1]=u
data[:,2]=k
data[:,3]=om
data[:,4]=vist
data[:,5]=uv
data[:,6]=yc
data[:,7]=C_mu
np.savetxt('yp_u_k_om_vist_uv_yc_PDH_5200.dat', data)

plt.show(block=True)
# #TODO testing CFD data
# dudy=np.minimum(dudy,dudy_max)
# dudy=np.maximum(dudy,dudy_min)





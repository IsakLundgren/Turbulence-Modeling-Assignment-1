import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
#import gradients.py
from gradients import compute_face_phi,dphidx,dphidy,init
plt.rcParams.update({'font.size': 22})

plt.interactive(True)

x = 2000

plt.close('all')

x=10

# read data file
tec=np.genfromtxt("tec.dat", dtype=None,comments="%")

#text='VARIABLES = X Y P U V u2 v2 w2 uv mu_sgs prod'

x=tec[:,0]
y=tec[:,1]
p=tec[:,2]
u=tec[:,3]
v=tec[:,4]
uu=tec[:,5]
vv=tec[:,6]
ww=tec[:,7]
uv=tec[:,8]
eps=tec[:,9]
k=0.5*(uu+vv+ww)

if max(y) == 1.:
   ni=170
   nj=194
   nu=1./10000.
else:
   nu=1./10595.
   if max(x) > 8.:
     nj=162
     ni=162
   else:
     ni=402
     nj=162

viscos=nu

u2d=np.reshape(u,(nj,ni))
v2d=np.reshape(v,(nj,ni))
p2d=np.reshape(p,(nj,ni))
x2d=np.reshape(x,(nj,ni))
y2d=np.reshape(y,(nj,ni))
uu2d=np.reshape(uu,(nj,ni)) #=mean{v'_1v'_1}
uv2d=np.reshape(uv,(nj,ni)) #=mean{v'_1v'_2}
vv2d=np.reshape(vv,(nj,ni)) #=mean{v'_2v'_2}
ww2d=np.reshape(ww,(nj,ni)) #=mean{v'_3v'_3}
k2d=np.reshape(k,(nj,ni))
eps2d=np.reshape(eps,(nj,ni))

u2d=np.transpose(u2d)
v2d=np.transpose(v2d)
p2d=np.transpose(p2d)
uu2d=np.transpose(uu2d)
vv2d=np.transpose(vv2d)
ww2d=np.transpose(ww2d)
uv2d=np.transpose(uv2d)
k2d=np.transpose(k2d)
eps2d=np.transpose(eps2d)


# set periodic b.c on west boundary
#u2d[0,:]=u2d[-1,:]
#v2d[0,:]=v2d[-1,:]
#p2d[0,:]=p2d[-1,:]
#uu2d[0,:]=uu2d[-1,:]


# read k and eps from a 2D RANS simulations. They should be used for computing the damping function f
k_eps_RANS = np.loadtxt("k_eps_RANS.dat")
k_RANS=k_eps_RANS[:,0]
eps_RANS=k_eps_RANS[:,1]
vist_RANS=k_eps_RANS[:,2]

ntstep=k_RANS[0]

k_RANS2d=np.reshape(k_RANS,(ni,nj))/ntstep
eps_RANS2d=np.reshape(eps_RANS,(ni,nj))/ntstep
vist_RANS2d=np.reshape(vist_RANS,(ni,nj))/ntstep

# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("mesh.dat")
xf=xc_yc[:,0]
yf=xc_yc[:,1]
x2d=np.reshape(xf,(nj-1,ni-1))
y2d=np.reshape(yf,(nj-1,ni-1))
x2d=np.transpose(x2d)
y2d=np.transpose(y2d)

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])

# compute geometric quantities
areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy = init(x2d,y2d,xp2d,yp2d)

# delete last row
u2d = np.delete(u2d, -1, 0)
v2d = np.delete(v2d, -1, 0)
p2d = np.delete(p2d, -1, 0)
k2d = np.delete(k2d, -1, 0)
uu2d = np.delete(uu2d, -1, 0)
vv2d = np.delete(vv2d, -1, 0)
ww2d = np.delete(ww2d, -1, 0)
uv2d = np.delete(uv2d, -1, 0)
eps2d = np.delete(eps2d, -1, 0)
k_RANS2d = np.delete(k_RANS2d, -1, 0)
eps_RANS2d = np.delete(eps_RANS2d, -1, 0)
vist_RANS2d = np.delete(vist_RANS2d, -1, 0)

# delete first row
u2d = np.delete(u2d, 0, 0)
v2d = np.delete(v2d, 0, 0)
p2d = np.delete(p2d, 0, 0)
k2d = np.delete(k2d, 0, 0)
uu2d = np.delete(uu2d, 0, 0)
vv2d = np.delete(vv2d, 0, 0)
ww2d = np.delete(ww2d, 0, 0)
uv2d = np.delete(uv2d, 0, 0)
eps2d = np.delete(eps2d, 0, 0)
k_RANS2d = np.delete(k_RANS2d, 0, 0)
eps_RANS2d = np.delete(eps_RANS2d, 0, 0)
vist_RANS2d = np.delete(vist_RANS2d, 0, 0)

# delete last columns
u2d = np.delete(u2d, -1, 1)
v2d = np.delete(v2d, -1, 1)
p2d = np.delete(p2d, -1, 1)
k2d = np.delete(k2d, -1, 1)
uu2d = np.delete(uu2d, -1, 1)
vv2d = np.delete(vv2d, -1, 1)
ww2d = np.delete(ww2d, -1, 1)
uv2d = np.delete(uv2d, -1, 1)
eps2d = np.delete(eps2d, -1, 1)
k_RANS2d = np.delete(k_RANS2d, -1, 1)
eps_RANS2d = np.delete(eps_RANS2d, -1, 1)
vist_RANS2d = np.delete(vist_RANS2d, -1, 1)

# delete first columns
u2d = np.delete(u2d, 0, 1)
v2d = np.delete(v2d, 0, 1)
p2d = np.delete(p2d, 0, 1)
k2d = np.delete(k2d, 0, 1)
uu2d = np.delete(uu2d, 0, 1)
vv2d = np.delete(vv2d, 0, 1)
ww2d = np.delete(ww2d, 0, 1)
uv2d = np.delete(uv2d, 0, 1)
eps2d = np.delete(eps2d, 0, 1)
k_RANS2d = np.delete(k_RANS2d, 0, 1)
eps_RANS2d = np.delete(eps_RANS2d, 0, 1)
vist_RANS2d = np.delete(vist_RANS2d, 0, 1)

ni = ni-2
nj = nj-2

# eps at last cell upper cell wrong. fix it.
eps2d[:,-1]=eps2d[:,-2]

# compute face value of U and V
u2d_face_w,u2d_face_s=compute_face_phi(u2d,fx,fy,ni,nj)
v2d_face_w,v2d_face_s=compute_face_phi(v2d,fx,fy,ni,nj)
UU2d_face_w,UU2d_face_s=compute_face_phi(u2d**2,fx,fy,ni,nj)
VV2d_face_w,VV2d_face_s=compute_face_phi(v2d**2,fx,fy,ni,nj)
UV2d_face_w,UV2d_face_s=compute_face_phi(np.multiply(u2d,v2d),fx,fy,ni,nj)
p2d_face_w,p2d_face_s=compute_face_phi(p2d,fx,fy,ni,nj)
uv2d_face_w,uv2d_face_s=compute_face_phi(uv2d,fx,fy,ni,nj) # reynolds components
uu2d_face_w,uu2d_face_s=compute_face_phi(uu2d,fx,fy,ni,nj) # Turbulent kinetic energy x
vv2d_face_w,vv2d_face_s=compute_face_phi(vv2d,fx,fy,ni,nj) # Turbulent kinetic energy y

# x derivatives
dudx=dphidx(u2d_face_w,u2d_face_s,areawx,areasx,vol)
dvdx=dphidx(v2d_face_w,v2d_face_s,areawx,areasx,vol)
dpdx=dphidx(p2d_face_w,p2d_face_s,areawx,areasx,vol)
dUUdx=dphidx(UU2d_face_w,UU2d_face_s,areawx,areasx,vol)
dUVdx=dphidx(UV2d_face_w,UV2d_face_s,areawx,areasx,vol)
duvdx = dphidx(uv2d_face_w,uv2d_face_s,areawx,areasx,vol) # reynolds stressesx-dir
duudx = dphidx(uu2d_face_w,uu2d_face_s,areawx,areasx,vol)# TKR gradient x-dir

# y derivatives
dudy=dphidy(u2d_face_w,u2d_face_s,areawy,areasy,vol)
dvdy=dphidy(v2d_face_w,v2d_face_s,areawy,areasy,vol)
dpdy=dphidy(p2d_face_w,p2d_face_s,areawy,areasy,vol)
dVVdy=dphidy(VV2d_face_w,VV2d_face_s,areawy,areasy,vol)
dUVdy=dphidy(UV2d_face_w,UV2d_face_s,areawy,areasy,vol)
duvdy = dphidy(uv2d_face_w,uv2d_face_s,areawy,areasy,vol) # reynolds stresses y-dir
dvvdy = dphidy(vv2d_face_w,vv2d_face_s,areawy,areasy,vol)# TKR gradient y-dir

# Face values of derivatives
dudx_w,dudx_s=compute_face_phi(dudx,fx,fy,ni,nj)
dvdx_w,dvdx_s=compute_face_phi(dvdx,fx,fy,ni,nj)
dudy_w,dudy_s=compute_face_phi(dudy,fx,fy,ni,nj)
dvdy_w,dvdy_s=compute_face_phi(dvdy,fx,fy,ni,nj)

# second derivatives
  #x
dudxdx = dphidx(dudx_w,dudx_s,areawx,areasx,vol)
dvdxdx = dphidx(dvdx_w,dvdx_s,areawx,areasx,vol)
  #y
dudydy = dphidy(dudy_w,dudy_s,areawy,areasy,vol)
dvdydy = dphidy(dvdy_w,dvdy_s,areawy,areasy,vol)

omega2d=eps2d/k2d/0.09

################################ mesh plot

fig = plt.figure()
for i in range(0, ni-1):
    plt.plot(x2d[i,:], y2d[i,:], 'k',linewidth=0.5)
    
    
for j in range(0, nj-1):
    plt.plot(x2d[:,j], y2d[:,j], 'k',linewidth=0.5)

plt.plot(x2d[0,:],y2d[0,:],'k-',linewidth=0.5)
plt.plot(x2d[-1,:],y2d[-1,:],'k-',linewidth=0.5)
plt.plot(x2d[:,0],y2d[:,0],'k-',linewidth=0.5)
plt.plot(x2d[:,-1],y2d[:,-1],'k-',linewidth=0.5)

plt.title('Mesh grid')
plt.savefig('grid.png')


################################ vector plot
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
k=6# plot every forth vector
ss=3.2 #vector length
plt.quiver(xp2d[::k,::k],yp2d[::k,::k],u2d[::k,::k],v2d[::k,::k],width=0.01)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("vector plot")
plt.savefig('vect_python.png')

################################ contour plot
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
plt.pcolormesh(xp2d,yp2d,dudy, vmin=-5,vmax=5,cmap=plt.get_cmap('hot'),shading='gouraud')
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r"the gradient $\partial \bar{v}_1/\partial x_2$")
plt.savefig('dudy.png')


#************
# plot uv
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
i=10
plt.plot(uv2d[i,:],yp2d[i,:],'b-')
plt.xlabel('$\overline{u^\prime v^\prime}$')
plt.ylabel('y/H')
plt.savefig('uv_python.png')

############
#Fresh code#
############

################################ Pressure gradient x-dir
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
plt.pcolormesh(xp2d,yp2d,dpdx)
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r"the gradient $\partial p/\partial x_1$")
plt.savefig('dpdx.png')

################################ Reynolds stress x-dir plot
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
plt.pcolormesh(xp2d,yp2d,duvdx)
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r"the gradient $\partial \bar{v_1'v_2'}/\partial x_1$")
plt.savefig('duvdx.png')

################################ Reynolds stress x-dir plot
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
plt.pcolormesh(xp2d,yp2d,duvdy)
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r"the gradient $\partial \bar{v_1'v_2'}/\partial x_2$")
plt.savefig('duvdy.png')

#Station plotting

i_close = 3 #Index close to the inlet (Might be too close)
i_circ = 30 #Index in circulating region

x_close = x[i_close]
x_circ = x[i_circ]

#Reynolds stress
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
i=10
plt.plot(uv2d[i_close,:],yp2d[i_close,:],'b-', label=('x = ' + str(x_close)))
plt.plot(uv2d[i_circ,:],yp2d[i_circ,:],'r-', label=('x = ' + str(x_circ)))
plt.xlabel('$\overline{u^\prime v^\prime}$')
plt.ylabel('y/H')
plt.legend()
plt.savefig('stationStress.png')

#Momentum components for the close case
plotloopivals = [i_close,i_circ]

for plotIteration in range(2):

  i = plotloopivals[plotIteration]

  #TODO clean plots
  fig2 = plt.figure()
  plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
  i=10
  plt.plot(dUUdx[i,:],yp2d[i,:], label='First convection term')
  plt.plot(dUVdy[i,:],yp2d[i,:], label='Second convection term')
  plt.plot(-dpdx[i,:],yp2d[i,:], label='Pressure gradient term')
  plt.plot(nu*dudxdx[i,:],yp2d[i,:], label='First viscous diffusion term')
  plt.plot(nu*dudydy[i,:],yp2d[i,:], label='Second viscous diffusion term')
  plt.plot(-duudx[i,:],yp2d[i,:], label='First turbulent diffusion term')
  plt.plot(-duvdy[i,:],yp2d[i,:], label='Second turbulent diffusion term (Reynolds stress gradient)')
  plt.title('x-momentum, x = ' + str(x_close))
  plt.xlabel('$\overline{u^\prime v^\prime}$')
  plt.ylabel('y/H')
  #plt.legend()
  plt.savefig('x-momentumClose.png')

  fig2 = plt.figure()
  plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
  i=10
  plt.plot(dUVdx[i,:],yp2d[i,:], label='First convection term')
  plt.plot(dVVdy[i,:],yp2d[i,:], label='Second convection term')
  plt.plot(-dpdy[i,:],yp2d[i,:], label='Pressure gradient term')
  plt.plot(nu*dvdxdx[i,:],yp2d[i,:], label='First viscous diffusion term')
  plt.plot(nu*dvdydy[i,:],yp2d[i,:], label='Second viscous diffusion term')
  plt.plot(-duvdx[i,:],yp2d[i,:], label='First turbulent diffusion term')
  plt.plot(-dvvdy[i,:],yp2d[i,:], label='Second turbulent diffusion term (Reynolds stress gradient)')
  plt.title('x-momentum, x = ' + str(x_close))
  plt.xlabel('$\overline{u^\prime v^\prime}$')
  plt.ylabel('y/H')
  #plt.legend()
  plt.savefig('x-momentumClose.png')

#Q1.4.5

#Choosing stress 11 and 12
#Setup turbulent viscosity
C_mu = 0.009
C_1 = 1.5
C_2 = 0.6
C_1w = 0.5
C_2w = 0.3
sigma_k = 1

nu_t = C_mu * np.divide(np.multiply(k_RANS2d,k_RANS2d),eps_RANS2d)

#Compute face values
Uuu_w,Uuu_s=compute_face_phi(np.multiply(u2d,uu2d),fx,fy,ni,nj)
Vuu_w,Vuu_s=compute_face_phi(np.multiply(v2d,uu2d),fx,fy,ni,nj)
Uuv_w,Uuv_s=compute_face_phi(np.multiply(u2d,uv2d),fx,fy,ni,nj)
Vuv_w,Vuv_s=compute_face_phi(np.multiply(v2d,uv2d),fx,fy,ni,nj)

#Compute first derivatives
  # x
dUuudx = dphidx(Uuu_w,Uuu_s,areawx,areasx,vol)
dUuvdx = dphidx(Uuv_w,Uuv_s,areawx,areasx,vol)

  # y
dVuudy = dphidy(Vuu_w,Vuu_s,areawy,areasy,vol)
dVuvdy = dphidy(Vuv_w,Vuv_s,areawy,areasy,vol)
duudy = dphidy(uu2d_face_w,uu2d_face_s,areawy,areasy,vol)

#Compute derivative face values
duudx_w,duudx_s = compute_face_phi(duudx,fx,fy,ni,nj)
duvdx_w,duvdx_s = compute_face_phi(duvdx,fx,fy,ni,nj)
duudy_w,duudy_s = compute_face_phi(duudy,fx,fy,ni,nj)
duvdy_w,duvdy_s = compute_face_phi(duvdy,fx,fy,ni,nj)
nu_t_w,nu_t_s = compute_face_phi(nu_t,fx,fy,ni,nj)

#Compute second derivatives
  #x
duudxdx = dphidx(duudx_w,duudx_s,areawx,areasx,vol)
duvdxdx = dphidx(duvdx_w,duvdx_s,areawx,areasx,vol)
dnu_tduudxdx = dphidx(np.multiply(nu_t_w,duudx_w),np.multiply(nu_t_s,duudx_s),areawx,areasx,vol)
dnu_tduvdxdx = dphidx(np.multiply(nu_t_w,duvdx_w),np.multiply(nu_t_s,duvdx_s),areawx,areasx,vol)

  #y
duudydy = dphidy(duudy_w,duudy_s,areawy,areasy,vol)
duvdydy = dphidy(duvdy_w,duvdy_s,areawy,areasy,vol)
dnu_tduudydy = dphidy(np.multiply(nu_t_w,duudy_w),np.multiply(nu_t_s,duudy_s),areawy,areasy,vol)
dnu_tduvdydy = dphidy(np.multiply(nu_t_w,duvdy_w),np.multiply(nu_t_s,duvdy_s),areawy,areasy,vol)

#Convective term
ConvectiveReynolds11 = dUuudx + dVuudy
ConvectiveReynolds12 = dUuvdx + dVuvdy

#Viscous diffusion term
ViscousReynolds11 = nu * (duudxdx + duudydy)
ViscousReynolds12 = nu * (duvdxdx + duvdydy)

#Produciton term
ProductionReynolds11 = - 2 * (np.multiply(uu2d,dudx) + np.multiply(uv2d,dudy))
ProductionReynolds12 = - (np.multiply(uu2d,dvdx) + np.multiply(uv2d,dvdy)) - (np.multiply(uv2d,dudx) + np.multiply(vv2d,dudy))

#Pressure-strain term
#Preperatory terms
ProductionTKE = -(np.multiply(uu2d,dudx) + np.multiply(uv2d,dudy) + np.multiply(uv2d,dvdx) + np.multiply(vv2d,dvdy)) #Turbulent Kinetic Energy

calcWallDist = False

if calcWallDist:
    Walldistance = np.zeros((ni,nj))
    for i in range(ni):
        for j in range(nj):
            Distances = np.zeros(ni)
            for k in range(ni):
                Distances[k] = np.sqrt((x2d[i,j]-x2d[k,1])**2 + (y2d[i,j]-y2d[k,1])**2)
            Walldistance[i,j] = np.min(Distances)
    
    # Write Walldistance vector to file
    np.savetxt("Walldistance.dat", Walldistance)
else:
    if os.path.isfile("Walldistance.dat"):
        # Read Walldistance vector from file
        Walldistance = np.loadtxt("Walldistance.dat")
    else:
        print("Error: Walldistance file not found.")

dampingFunction = np.zeros((ni,nj))
for i in range(ni):
   for j in range(nj):
      dampingFunction[i,j] = min(k_RANS2d[i,j]**(3/2)/(2.55*Walldistance[i,j]*eps_RANS2d[i,j]),1)

SPR11 = - C_1 * np.multiply(np.divide(eps_RANS2d,k_RANS2d),(uu2d - 2/3 * k_RANS2d))
RPR11 = - C_2 * (ProductionReynolds11 - 2/3 * ProductionTKE)
SPR12 = - C_1 * np.multiply(np.divide(eps_RANS2d,k_RANS2d),uv2d)
RPR12 = - C_2 * ProductionReynolds12

SPWR11 = C_1w * np.multiply(np.divide(eps_RANS2d,k_RANS2d),vv2d,dampingFunction)
RPWR11 = C_2w * np.multiply(RPR11,dampingFunction)
SPWR12 = - 3/2 * C_1w * np.multiply(np.divide(eps_RANS2d,k_RANS2d),uv2d,dampingFunction)
RPWR12 = - 3/2 * C_1w * np.multiply(RPR12,dampingFunction)

PressureStrainReynolds11 = SPR11 + RPR11 + SPWR11 + RPWR11
PressureStrainReynolds12 = SPR12 + RPR12 + SPWR12 + RPWR12

#Turbulent diffusion term
TurbulentReynolds11 = 1/sigma_k * (dnu_tduudxdx + dnu_tduudydy)
TurbulentReynolds12 = 1/sigma_k * (dnu_tduvdxdx + dnu_tduvdydy)

#Destruction term
DestructionReynolds11 = -2/3 * eps_RANS2d
DestructionReynolds12 = -np.zeros((ni,nj))

fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
i=i_close
plt.plot(ConvectiveReynolds11[i,:],yp2d[i,:],label='Convective term')
plt.plot(ViscousReynolds11[i,:],yp2d[i,:],label='Viscous diffusion term')
plt.plot(ProductionReynolds11[i,:],yp2d[i,:],label='Production term')
plt.plot(PressureStrainReynolds11[i,:],yp2d[i,:],label='Pressure strain term')
plt.plot(TurbulentReynolds11[i,:],yp2d[i,:],label='Turbulent diffusion term')
plt.plot(DestructionReynolds11[i,:],yp2d[i,:],label='Destruction term')

plt.title('Reynolds 11 equation terms at x = ' + str(x_close))
plt.xlabel('$Equation term magnitude [m^2s^{-3}]$')
plt.ylabel('y/H')
plt.legend()
plt.savefig('Reynolds11.png')

fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
i=i_close
plt.plot(ConvectiveReynolds12[i,:],yp2d[i,:],label='Convective term')
plt.plot(ViscousReynolds12[i,:],yp2d[i,:],label='Viscous diffusion term')
plt.plot(ProductionReynolds12[i,:],yp2d[i,:],label='Production term')
plt.plot(PressureStrainReynolds12[i,:],yp2d[i,:],label='Pressure strain term')
plt.plot(TurbulentReynolds12[i,:],yp2d[i,:],label='Turbulent diffusion term')
plt.plot(DestructionReynolds12[i,:],yp2d[i,:],label='Destruction term')

plt.title('Reynolds 12 equation terms at x = ' + str(x_close))
plt.xlabel('$Equation term magnitude [m^2s^{-3}]$')
plt.ylabel('y/H')
plt.legend()
plt.savefig('Reynolds12.png')

#Q1.4.6

#B for Boussinesq

uuB = -np.multiply(nu_t,(dudx + dudx)) + 2/3 * k_RANS2d
uvB = -np.multiply(nu_t,(dudy + dvdx))
vuB = uvB
vvB = -np.multiply(nu_t,(dvdy + dvdy)) + 2/3 * k_RANS2d

NormDiff11 = 2 * np.divide((uu2d-uuB),(uu2d+uuB))
NormDiff12 = 2 * np.divide((uv2d-uvB),(uv2d+uvB))

fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
plt.pcolormesh(xp2d,yp2d,NormDiff11)
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r"Normalized difference 11 term")
plt.savefig('BoussinesqComp11.png')

fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
plt.pcolormesh(xp2d,yp2d,NormDiff12)
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r"Normalized difference 12 term")
plt.savefig('BoussinesqComp12.png')


#Q1.4.7

FindNegativesPTKE = np.sign(ProductionTKE)

################################ Pressure gradient x-dir
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
plt.pcolormesh(xp2d,yp2d,FindNegativesPTKE)
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r"Sign function of the exact TKE production term")
plt.savefig('PTKENeg.png')

#Q1.4.8

#Calculate Eigenvalues in each cell

s11 = dudx
s12 = 1/2 * (dudy + dvdx)
s21 = s12
s22 = dvdx

SEig= np.zeros(shape = (ni,nj,2))
nu_tReduction = np.zeros([ni,nj])

for i in range(ni):
   for j in range(nj):
      SEig[i,j,:] = np.linalg.eigvals([[s11[i,j],s12[i,j]],[s21[i,j],s22[i,j]]])
      nu_tReduction = nu_t > np.max(SEig)

#TODO plotting
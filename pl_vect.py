"""
Header
"""

import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
#import gradients.py
from gradients import compute_face_phi,dphidx,dphidy,init
plt.rcParams.update({'font.size': 22})
import matplotlib.ticker as mtick
from matplotlib import ticker

plt.interactive(True)

plt.close('all')

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

k_RANS2d=np.reshape(k_RANS,(nj,ni))/ntstep
eps_RANS2d=np.reshape(eps_RANS,(nj,ni))/ntstep
vist_RANS2d=np.reshape(vist_RANS,(nj,ni))/ntstep

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
p2d_face_w,p2d_face_s=compute_face_phi(p2d,fx,fy,ni,nj)
uu2d_face_w,uu2d_face_s=compute_face_phi(uu2d,fx,fy,ni,nj)
uv2d_face_w,uv2d_face_s=compute_face_phi(uv2d,fx,fy,ni,nj)

# x derivatives
dudx=dphidx(u2d_face_w,u2d_face_s,areawx,areasx,vol)
dvdx=dphidx(v2d_face_w,v2d_face_s,areawx,areasx,vol)
dpdx=dphidx(p2d_face_w,p2d_face_s,areawx,areasx,vol)
duudx=dphidx(uu2d_face_w,uu2d_face_s,areawx,areasx,vol)
duvdx=dphidx(uv2d_face_w,uv2d_face_s,areawx,areasx,vol)

# y derivatives
dudy=dphidx(u2d_face_w,u2d_face_s,areawy,areasy,vol)
dvdy=dphidx(v2d_face_w,v2d_face_s,areawy,areasy,vol)
dpdy=dphidx(p2d_face_w,p2d_face_s,areawy,areasy,vol)
duudy=dphidx(uu2d_face_w,uu2d_face_s,areawy,areasy,vol)
duvdy=dphidx(uv2d_face_w,uv2d_face_s,areawy,areasy,vol)

omega2d=eps2d/k2d/0.09

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


# #************
# # plot uv
# fig2 = plt.figure()
# plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
# i=10
# plt.plot(uv2d[i,:],yp2d[i,:],'b-')
# plt.xlabel('$\overline{u^\prime v^\prime}$')
# plt.ylabel('y/H')
# plt.savefig('uv_python.png')

#************
# plot uv
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i=10
ax1.plot(uv2d[i,:],yp2d[i,:],'b-',label='i=1')
i=30
ax1.plot(uv2d[i,:],yp2d[i,:],'r-',label='i=30')
M = 2 # use only two major xticks
xticks = ticker.MaxNLocator(M)
ax1.xaxis.set_major_locator(xticks)
plt.legend(loc='best',prop=dict(size=13))
#ax1.yaxis.set_label_coords(-0.095, 0.6)
plt.xlabel('$\overline{u^\prime v^\prime}$')
plt.ylabel('y/H')
plt.savefig('uv_python.png')





# Own plots

### R1.1  PLOTS OF REYNOLD SHEAR STRESS force per unit volume
# contour plot duvdx
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
plt.pcolormesh(xp2d,yp2d, duvdx)#, vmin=-5,vmax=5,cmap=plt.get_cmap('hot'),shading='gouraud')
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y/H$")
plt.title(r"the gradient $\partial \bar{v_1v_2}/\partial x_1$")
plt.savefig('duvdx.png')

# contour plot duvdy
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
plt.pcolormesh(xp2d,yp2d, duvdy)#, vmin=-5,vmax=5,cmap=plt.get_cmap('hot'),shading='gouraud')
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y/H$")
plt.title(r"the gradient $\partial \bar{v\prime _1v\prime _2}/\partial x_2$")
plt.savefig('duvdy.png')


### R1.2, 1.1
# Plot the stresses along two x1-locaitons
i1 = 1; i2 = 20
fig3 = plt.figure()
plt.plot(duvdx[i1,:], yp2d[i1,:], label='x1=1') 
plt.plot(duvdx[i2,:], yp2d[i2,:], label='x1=20')
plt.xlabel(r"$\partial u{prime v\prime / \partial x$")
plt.legend()
plt.title(r'Shearstress at 2 x1')

### R1.2,  1.2
# plot all forces in the RANS eq.
# duu_bar_dx, d2u_dx2, d2u_dy2, du_pm_dx (=dmean(u')^2_dx), dv_pm_dy (=dmean(v')^2_dy)

# convective components: e.g. mean(u*u)
duu_bar_dx = 2 * u2d*dudx 
duv_bar_dx = u2d*dvdx + v2d*dudx
duv_bar_dy = u2d*dvdy + v2d*dudy
dvv_bar_dy = 2 * v2d*dvdy

# viscous components
dudx2d_face_w,dudx2d_face_s=compute_face_phi(dudx,fx,fy,ni,nj)
dvdx2d_face_w,dvdx2d_face_s=compute_face_phi(dvdx,fx,fy,ni,nj)
dudy2d_face_w,dudy2d_face_s=compute_face_phi(dudy,fx,fy,ni,nj)
dvdy2d_face_w,dvdy2d_face_s=compute_face_phi(dvdy,fx,fy,ni,nj)

d2u_dx2 = dphidx(dudx2d_face_w, dudx2d_face_s, areawx, areasx, vol)
d2v_dx2 = dphidx(dvdx2d_face_w, dvdx2d_face_s, areawx, areasx, vol)
d2u_dy2 = dphidx(dudy2d_face_w, dudy2d_face_s, areawy, areasy, vol)
d2v_dy2 = dphidx(dvdy2d_face_w, dvdy2d_face_s, areawy, areasy, vol)

# Plots of the component in the v_bar_1 equation
fig4=plt.figure()
plt.plot(duu_bar_dx[i1,:],  yp2d[i1,:], label='duu_bar_dx')
plt.plot(duv_bar_dx[i1,:],  yp2d[i1,:], label='duv_bar_dx')
plt.plot(duv_bar_dy[i1,:],  yp2d[i1,:], label='duv_bar_dy')
plt.plot(dvv_bar_dy[i1,:],  yp2d[i1,:], label='dvv_bar_dy') # could prob omitt

fig5 = plt.figure() # Components in order as equaiton
plt.plot(dpdx[i1,:], yp2d[i1,:], label='dpdx')
plt.plot(nu*d2u_dx2[i1,:], yp2d[i1,:], label='nu*d2u_dx2')
plt.plot(duudx[i1,:], yp2d[i1,:], label='duu_dx')
plt.plot(nu*d2u_dy2[i1,:], yp2d[i1,:], label='nu*d2u_dy2')
plt.plot(duvdy[i1,:], yp2d[i1,:], label='duv_dy')
plt.legend()


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
#ax1.semilogx(data[:,1],data[:,2])
plt.plot(dpdx[i1,:], yp2d[i1,:], label='dpdx')
plt.plot(nu*d2u_dx2[i1,:], yp2d[i1,:], label='nu*d2u_dx2')
plt.plot(duudx[i1,:], yp2d[i1,:], label='duu_dx')
plt.plot(nu*d2u_dy2[i1,:], yp2d[i1,:], label='nu*d2u_dy2')
plt.plot(duvdy[i1,:], yp2d[i1,:], label='duv_dy')

ax2 = plt.axes([.65, .6, .2, .2], axisbg='y')
plt.plot(dpdx[i1,0:0.2], yp2d[i1,0:0.2], label='dpdx')
plt.plot(nu*d2u_dx2[i1,0:0.2], yp2d[i1,0:0.2], label='nu*d2u_dx2')
plt.plot(duudx[i1,0:0.2], yp2d[i1,0:0.2], label='duu_dx')
plt.plot(nu*d2u_dy2[i1,0:0.2], yp2d[i1,0:0.2], label='nu*d2u_dy2')
plt.plot(duvdy[i1,0:0.2], yp2d[i1,0:0.2], label='duv_dy')
plt.setp(ax2, xticks=[], yticks=[])
plt.show




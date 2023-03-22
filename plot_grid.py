import scipy.io as sio
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 22})


plt.close('all')

# read data file
tec=np.genfromtxt("tec.dat", dtype=None,comments="%")

#text='VARIABLES = X Y P U V u2 v2 w2 uv mu_sgs prod'

x=tec[:,0]
y=tec[:,1]

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


# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("mesh.dat")
xf=xc_yc[:,0]
yf=xc_yc[:,1]
x2d=np.reshape(xf,(nj-1,ni-1))
y2d=np.reshape(yf,(nj-1,ni-1))
x2d=np.transpose(x2d)
y2d=np.transpose(y2d)


#%%%%%%%%%%%%%%%%%%%%% grid
fig2 = plt.figure()
for i in range (0,ni-1):
   plt.plot(x2d[i,:],y2d[i,:])

for j in range (0,nj-1):
   plt.plot(x2d[:,j],y2d[:,j])

#plt.axis([0,5,0,5])
plt.title('grid.png')
plt.savefig('grid.png')


#%%%%%%%%%%%%%%%%%%%%% grid contour
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(x2d[0,:],y2d[0,:],'b-',linewidth=3)
plt.plot(x2d[-1,:],y2d[-1,:],'b-',linewidth=3)
plt.plot(x2d[:,0],y2d[:,0],'b-',linewidth=3)
plt.plot(x2d[:,-1],y2d[:,-1],'b-',linewidth=3)
plt.xlabel("$x$")
plt.ylabel("$y$")

#plt.axis([0,5,0,5])
plt.savefig('domain.eps')

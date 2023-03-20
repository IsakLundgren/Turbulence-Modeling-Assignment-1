def init(x2d,y2d,xp2d,yp2d):
   import numpy as np

#  west face coordinate
   xw=0.5*(x2d[0:-1,0:-1]+x2d[0:-1,1:])
   yw=0.5*(y2d[0:-1,0:-1]+y2d[0:-1,1:])

   del1x=((xw-xp2d)**2+(yw-yp2d)**2)**0.5
   del2x=((xw-np.roll(xp2d,1,axis=0))**2+(yw-np.roll(yp2d,1,axis=0))**2)**0.5
   fx=del2x/(del1x+del2x)
#  if cyclic_x:
   fx[0,:]=0.5

#  south face coordinate
   xs=0.5*(x2d[0:-1,0:-1]+x2d[1:,0:-1])
   ys=0.5*(y2d[0:-1,0:-1]+y2d[1:,0:-1])

   del1y=((xs-xp2d)**2+(ys-yp2d)**2)**0.5
   del2y=((xs-np.roll(xp2d,1,axis=1))**2+(ys-np.roll(yp2d,1,axis=1))**2)**0.5
   fy=del2y/(del1y+del2y)

   areawy=np.diff(x2d,axis=1)
   areawx=-np.diff(y2d,axis=1)

   areasy=-np.diff(x2d,axis=0)
   areasx=np.diff(y2d,axis=0)

   areaw=(areawx**2+areawy**2)**0.5
   areas=(areasx**2+areasy**2)**0.5

# volume approaximated as the vector product of two triangles for cells
   ax=np.diff(x2d,axis=1)
   ay=np.diff(y2d,axis=1)
   bx=np.diff(x2d,axis=0)
   by=np.diff(y2d,axis=0)

   areaz_1=0.5*np.absolute(ax[0:-1,:]*by[:,0:-1]-ay[0:-1,:]*bx[:,0:-1])

   ax=np.diff(x2d,axis=1)
   ay=np.diff(y2d,axis=1)
   areaz_2=0.5*np.absolute(ax[1:,:]*by[:,0:-1]-ay[1:,:]*bx[:,0:-1])

   vol=areaz_1+areaz_2

   return areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy

def compute_face_phi(phi2d,fx,fy,ni,nj):
   import numpy as np

   phi2d_face_w=np.empty((ni+1,nj))
   phi2d_face_s=np.empty((ni,nj+1))
   phi2d_face_w[0:-1,:]=fx*phi2d+(1-fx)*np.roll(phi2d,1,axis=0)
   phi2d_face_s[:,0:-1]=fy*phi2d+(1-fy)*np.roll(phi2d,1,axis=1)

# west boundary 
#  cyclic_x:
   phi2d_face_w[0,:]=0.5*(phi2d[0,:]+phi2d[-1,:])

# east boundary 
#  cyclic_x:
   phi2d_face_w[-1,:]=0.5*(phi2d[0,:]+phi2d[-1,:])


# south boundary 
   phi2d_face_s[:,0]=0

# north boundary 
# neumann
   phi2d_face_s[:,-1]=phi2d[:,-1]
   
   return phi2d_face_w,phi2d_face_s

def dphidx(phi_face_w,phi_face_s,areawx,areasx,vol):

   phi_w=phi_face_w[0:-1,:]*areawx[0:-1,:]
   phi_e=-phi_face_w[1:,:]*areawx[1:,:]
   phi_s=phi_face_s[:,0:-1]*areasx[:,0:-1]
   phi_n=-phi_face_s[:,1:]*areasx[:,1:]
   return (phi_w+phi_e+phi_s+phi_n)/vol

def dphidy(phi_face_w,phi_face_s,areawy,areasy,vol):

   phi_w=phi_face_w[0:-1,:]*areawy[0:-1,:]
   phi_e=-phi_face_w[1:,:]*areawy[1:,:]
   phi_s=phi_face_s[:,0:-1]*areasy[:,0:-1]
   phi_n=-phi_face_s[:,1:]*areasy[:,1:]
   return (phi_w+phi_e+phi_s+phi_n)/vol


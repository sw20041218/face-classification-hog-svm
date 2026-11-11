import numpy as np
from scipy.ndimage import gaussian_filter
from face_model import unit,contrast,hog,flip

def shift(a,dy,dx):
 im=a.reshape(-1,19,19).transpose(0,2,1)
 p=np.pad(im,((0,0),(1,1),(1,1)),mode='edge')
 return p[:,1-dy:20-dy,1-dx:20-dx].transpose(0,2,1).reshape(-1,361).copy()

def local_contrast(x):
 im=x.reshape(-1,19,19)
 mu=gaussian_filter(im,(0,1.5,1.5),mode='reflect')
 var=np.maximum(gaussian_filter(im*im,(0,1.5,1.5),mode='reflect')-mu*mu,0)
 z=(im-mu)/np.sqrt(var+0.03**2)
 return unit(np.clip(z,-3,3).reshape(-1,361))

def coarse_hog(x):
 im=x.reshape(-1,19,19).transpose(0,2,1);gy,gx=np.gradient(im,axis=(1,2))
 mag=np.hypot(gx,gy);angle=np.mod(np.arctan2(gy,gx),np.pi)*9/np.pi
 lo=np.floor(angle).astype(int)%9;frac=angle-np.floor(angle)
 cells=np.zeros((len(x),2,2,9))
 for r in range(2):
  for c in range(2):
   sl=(slice(None),slice(1+r*8,9+r*8),slice(1+c*8,9+c*8))
   for b in range(9):cells[:,r,c,b]=np.sum(mag[sl]*((lo[sl]==b)*(1-frac[sl])+((lo[sl]+1)%9==b)*frac[sl]),axis=(1,2))
 return unit(np.minimum(unit(cells.reshape(len(x),36)),.2))

def branches(x):return contrast(x),hog(x),local_contrast(x),coarse_hog(x)
def combine(bs,mode,w):
 z,h,l,c=bs;parts=[z,h]
 if mode in ['A','C']:parts.append(w*l)
 if mode in ['B','C']:parts.append(w*c)
 return unit(np.concatenate(parts,axis=1))
def transform(x,mode,w):return combine(branches(x),mode,w)

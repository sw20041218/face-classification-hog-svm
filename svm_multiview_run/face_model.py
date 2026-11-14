"""19x19 face/non-face classifier. No network access or training on import."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

def load_data(path, key=None):
    data = loadmat(path)
    if key is not None:
        if key not in data:
            raise ValueError(f'Variable {key!r} not found; available: {list(data)}')
        x = data[key]
    else:
        candidates = [(k,v) for k,v in data.items() if not k.startswith('__') and
                      isinstance(v,np.ndarray) and v.ndim == 2 and 361 in v.shape]
        if len(candidates) != 1:
            raise ValueError(f'Expected one matrix with 361 features; found {[k for k,v in candidates]}; specify --key')
        key,x = candidates[0]
    x = np.asarray(x,dtype=np.float64)
    if x.ndim != 2: raise ValueError('Expected a matrix')
    if x.shape[1] != 361 and x.shape[0] == 361: x=x.T
    if x.shape[1] != 361 or len(x)==0 or not np.isfinite(x).all():
        raise ValueError(f'Invalid input shape or values: {x.shape}')
    if x.min() < 0 or x.max() > 255: raise ValueError('Pixels must be in [0,1] or [0,255]')
    if x.max() > 1: x=x/255.
    return x,key

def flip(x):
    # MATLAB image(:)' stores columns; row-major reshape is the transposed image.
    return np.asarray(x).reshape(-1,19,19)[:,::-1,:].reshape(-1,361).copy()

def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def contrast(x):
    return unit(x-x.mean(axis=1,keepdims=True))

def hog(x):
    images=x.reshape(-1,19,19).transpose(0,2,1)
    gy,gx=np.gradient(images,axis=(1,2))
    mag=np.hypot(gx,gy)
    angle=np.mod(np.arctan2(gy,gx),np.pi)*9/np.pi
    lower=np.floor(angle).astype(int)%9
    fraction=angle-np.floor(angle)
    cells=np.zeros((len(x),4,4,9))
    for r in range(4):
        for c in range(4):
            sl=(slice(None),slice(r*4+1,r*4+5),slice(c*4+1,c*4+5))
            for b in range(9):
                cells[:,r,c,b]=np.sum(mag[sl]*((lower[sl]==b)*(1-fraction[sl])+((lower[sl]+1)%9==b)*fraction[sl]),axis=(1,2))
    blocks=[]
    for r in range(3):
        for c in range(3):
            block=unit(cells[:,r:r+2,c:c+2,:].reshape(len(x),-1))
            blocks.append(unit(np.minimum(block,.2)))
    return unit(np.concatenate(blocks,axis=1))

def features(x, mode):
    if mode=='raw': return unit(x)
    z=contrast(x)
    if mode=='contrast': return z
    if mode.startswith('hog_'):
        w=float(mode.split('_')[1]);return unit(np.concatenate([z,w*hog(x)],axis=1))
    if mode=='smooth':
        im=x.reshape(-1,19,19)
        smooth=gaussian_filter(im,sigma=(0,.8,.8),mode='reflect').reshape(-1,361)
        return unit(np.concatenate([z,contrast(smooth)],axis=1))
    if mode=='crop':
        center=x.reshape(-1,19,19)[:,2:17,2:17].reshape(-1,225)
        return unit(np.concatenate([z,contrast(center)],axis=1))
    raise ValueError(mode)

def best_threshold(scores,y):
    scores=np.asarray(scores);y=np.asarray(y)
    values=np.unique(scores)
    thresholds=np.r_[np.nextafter(values[0],-np.inf),(values[1:]+values[:-1])/2,np.nextafter(values[-1],np.inf),0.]
    correct=np.sum((scores[:,None]>thresholds[None,:])==(y[:,None]==1),axis=0)
    best=np.flatnonzero(correct==correct.max())
    index=best[np.argmin(np.abs(thresholds[best]))]
    return float(thresholds[index]),int(correct[index])

def metrics(y,pred):
    return dict(n=int(len(y)),correct=int(np.sum(y==pred)),accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),f1_face=float(f1_score(y,pred,pos_label=1)),
        confusion_labels=[-1,1],confusion_matrix=confusion_matrix(y,pred,labels=[-1,1]).tolist())

class FaceMember:
    def __init__(self,config): self.config=config
    def fit(self,x,y,domain):
        cfg=self.config
        weight=np.where(domain==1,cfg['important_weight'],1.)*np.where(y==1,cfg['positive_weight'],1.)
        if cfg.get('augment',False):
            x=np.concatenate([x,flip(x)]);y=np.tile(y,2);weight=np.tile(weight,2)/2
        self.svc=SVC(C=cfg['C'],gamma=cfg['gamma'],kernel='rbf',cache_size=256,tol=1e-4,max_iter=100000)
        self.svc.fit(features(x,cfg['mode']),y,sample_weight=weight)
        if self.svc.fit_status_ != 0: raise RuntimeError('Final SVM did not converge')
        return self
    def decision_function(self,x):
        cfg=self.config
        score=self.svc.decision_function(features(x,cfg['mode']))
        if cfg.get('augment',False): score=(score+self.svc.decision_function(features(flip(x),cfg['mode'])))/2
        return score

class FaceEnsemble:
    def __init__(self,members,weights,scales,threshold):
        self.members=members;self.weights=np.asarray(weights);self.scales=np.asarray(scales);self.threshold=threshold
    def decision_function(self,x):
        scores=np.column_stack([m.decision_function(x) for m in self.members])
        return (scores/self.scales)@self.weights
    def predict(self,x): return np.where(self.decision_function(x)>self.threshold,1,-1).astype(np.int16)

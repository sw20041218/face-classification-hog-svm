from pathlib import Path
import json,time,hashlib
import numpy as np,joblib
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from face_model import features,flip
r=Path(__file__).parent.parent;o=Path(__file__).parent
positions=[(0,0),(-1,0),(1,0),(0,-1),(0,1)]
def shift(a,dy,dx):
 im=a.reshape(-1,19,19).transpose(0,2,1);p=np.pad(im,((0,0),(1,1),(1,1)),mode='edge')
 return p[:,1-dy:20-dy,1-dx:20-dx].transpose(0,2,1).reshape(-1,361).copy()
def views(x):
 result=[]
 for dy,dx in positions:
  a=shift(x,dy,dx);result.extend([a,flip(a)])
 return result
def score(m,fs):return np.mean([m.decision_function(f) for f in fs],axis=0)
x=loadmat(r/'train(Task_2)'/'train_data.mat')['train_data'];y=loadmat(r/'train(Task_2)'/'train_label.mat')['train_label'].ravel().astype(int)
assert x.shape==(800,361)
fs=[features(a,'hog_1.0') for a in views(x)];f=np.concatenate(fs);yf=np.tile(y,10)
folds=list(StratifiedKFold(5,shuffle=True,random_state=20261110).split(x,y));logs=[];start=time.time()
for fold,(tr,va) in enumerate(folds,1):
 idx=np.concatenate([tr+len(x)*k for k in range(10)])
 m=SVC(C=1.,gamma=2.,kernel='rbf',tol=1e-4,max_iter=100000,cache_size=512)
 m.fit(f[idx],yf[idx],sample_weight=np.full(len(idx),.1));assert m.fit_status_==0
 s=score(m,[a[va] for a in fs]);base=score(m,[fs[0][va],fs[1][va]])
 logs.append(dict(fold=fold,n=len(va),multiview_acc=float(np.mean((s>0)==(y[va]==1))),original_mirror_acc=float(np.mean((base>0)==(y[va]==1)))))
 print(json.dumps(logs[-1]),flush=True)
# Reuse frozen 0.910 model: isolate inference change, no test-driven fitting.
mp=r/'svm_shift_run'/'model.joblib';bundle=joblib.load(mp);m=bundle['svc'];assert bundle['config']==dict(C=1.,gamma=2.,face_weight=1.)
t=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data'];assert t.shape==(420,361)
s=score(m,[features(a,'hog_1.0') for a in views(t)]);pred=np.where(s>0,1,-1)
p=o/'result_multiview.txt';p.write_bytes(''.join(f'{i} {int(v)}\n' for i,v in enumerate(pred,1)).encode('ascii'))
a=np.loadtxt(p,dtype=int);assert a.shape==(420,2) and np.array_equal(a[:,0],np.arange(1,421)) and np.array_equal(a[:,1],pred)
basepred=np.loadtxt(r/'svm_shift_run'/'result_shift_svm.txt',dtype=int)[:,1]
report=dict(protocol='Fixed prior C=1 gamma=2; 5-fold training-only validation; ten augmentation views grouped by original; no test labels read; mean of ten view decision scores; threshold 0; no model ensemble',folds=logs,cv_mean=float(np.mean([a['multiview_acc'] for a in logs])),cv_std=float(np.std([a['multiview_acc'] for a in logs])),model_source=str(mp),model_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),prediction_counts={'face':int(sum(pred==1)),'nonface':int(sum(pred==-1))},changed_vs_0910=int(sum(pred!=basepred)),changed_ids=(np.flatnonzero(pred!=basepred)+1).tolist(),runtime_seconds=time.time()-start)
(o/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('DONE',json.dumps(report),flush=True)

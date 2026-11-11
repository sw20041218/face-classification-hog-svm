from pathlib import Path
import json,time,hashlib
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
import joblib
from face_model import features,flip
r=Path(__file__).parent.parent;o=Path(__file__).parent
x=loadmat(r/'train(Task_2)'/'train_data.mat')['train_data'];y=loadmat(r/'train(Task_2)'/'train_label.mat')['train_label'].ravel().astype(int)
assert x.shape==(800,361) and set(y)=={-1,1}
# Restore MATLAB orientation, shift with edge padding, then flatten in column order.
def shift(a,dy,dx):
 im=a.reshape(-1,19,19).transpose(0,2,1)
 pad=np.pad(im,((0,0),(1,1),(1,1)),mode='edge')
 return pad[:,1-dy:20-dy,1-dx:20-dx].transpose(0,2,1).reshape(-1,361).copy()
variants=[]
for dy,dx in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
 a=shift(x,dy,dx);variants.extend([a,flip(a)])
# Per-image deterministic features may be cached before splitting; no fitted transform.
f=np.concatenate([features(a,'hog_1.0') for a in variants])
orig=features(x,'hog_1.0');mir=features(flip(x),'hog_1.0');yf=np.tile(y,10)
folds=list(StratifiedKFold(5,shuffle=True,random_state=20261110).split(x,y))
configs=[dict(C=c,gamma=g,face_weight=w) for c in [1.,5.,20.] for g in [.5,2.] for w in [1.,1.5,2.]]
scores={str(cfg):np.empty(800) for cfg in configs};logs=[];start=time.time()
# Reuse each fold's kernel among C/class-weight candidates; no test data involved.
for fold,(tr,va) in enumerate(folds,1):
 aug=np.concatenate([tr+800*k for k in range(10)]);a=f[aug];yy=yf[aug]
 for gamma in [.5,2.]:
  kt=rbf_kernel(a,a,gamma=gamma);kv=(rbf_kernel(orig[va],a,gamma=gamma)+rbf_kernel(mir[va],a,gamma=gamma))/2
  for cfg in [c for c in configs if c['gamma']==gamma]:
   weights=np.where(yy==1,cfg['face_weight'],1.)/10
   m=SVC(C=cfg['C'],kernel='precomputed',tol=1e-4,max_iter=100000,cache_size=512)
   m.fit(kt,yy,sample_weight=weights)
   if m.fit_status_!=0:raise RuntimeError('SVM did not converge')
   score=m.decision_function(kv);scores[str(cfg)][va]=score
   log=dict(fold=fold,config=cfg,accuracy=float(accuracy_score(y[va],np.where(score>0,1,-1))))
   logs.append(log)
  del kt,kv
  print('Completed fold',fold,'gamma',gamma,'elapsed',round(time.time()-start,1),flush=True)
 (o/'cv_progress.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
trials=[]
for cfg in configs:
 ss=scores[str(cfg)];acc=[float(np.mean((ss[va]>0)==(y[va]==1))) for tr,va in folds]
 trials.append(dict(config=cfg,fold_accuracy=acc,cv_mean=float(np.mean(acc)),cv_std=float(np.std(acc))))
best=max(trials,key=lambda a:a['cv_mean']);cfg=best['config']
# Final single model trained on all 800 originals and ten variants, total weight per original = class weight.
m=SVC(C=cfg['C'],gamma=cfg['gamma'],kernel='rbf',tol=1e-4,max_iter=100000,cache_size=768)
m.fit(f,yf,sample_weight=np.where(yf==1,cfg['face_weight'],1.)/10)
assert m.fit_status_==0
joblib.dump(dict(svc=m,mode='hog_1.0',config=cfg,threshold=0.,mirror_tta=True),o/'model.joblib')
# Only now load unlabeled test inputs. Never read any test labels or annotation files.
t=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data'];assert t.shape==(420,361)
ss=(m.decision_function(features(t,'hog_1.0'))+m.decision_function(features(flip(t),'hog_1.0')))/2
pred=np.where(ss>0,1,-1)
result=''.join(f'{i} {int(v)}\n' for i,v in enumerate(pred,1));(o/'result_shift_svm.txt').write_bytes(result.encode('ascii'))
rows=(o/'result_shift_svm.txt').read_text(encoding='ascii').splitlines()
assert len(rows)==420 and all(line==f'{i} {int(pred[i-1])}' for i,line in enumerate(rows,1))
report=dict(protocol='Train-only stratified 5-fold selection; no test labels read; augmentation only within each fold; equal per-original total augmentation weight; threshold 0; original/mirror mean decision scores; ties resolved by predefined config order',seed=20261110,augmentation='original and 4 one-pixel edge-padded translations, each with horizontal mirror; 10 variants',training_originals=800,training_augmented=8000,best=best,trials=trials,test_prediction_counts={'face':int(sum(pred==1)),'nonface':int(sum(pred==-1))},runtime_seconds=time.time()-start)
(o/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('DONE',json.dumps(report),flush=True)

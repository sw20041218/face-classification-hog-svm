from pathlib import Path
import time,json
import numpy as np,joblib
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import rbf_kernel
from model_features import branches,combine,transform,shift,flip
r=Path(__file__).parent.parent;o=Path(__file__).parent
x=loadmat(r/'train(Task_2)'/'train_data.mat')['train_data'];y=loadmat(r/'train(Task_2)'/'train_label.mat')['train_label'].ravel().astype(int)
assert x.shape==(800,361) and set(y)=={-1,1}
variants=[]
for dy,dx in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
 a=shift(x,dy,dx);variants.extend([a,flip(a)])
b=branches(np.concatenate(variants));bo=branches(x);bm=branches(flip(x));yf=np.tile(y,10)
folds=list(StratifiedKFold(5,shuffle=True,random_state=20261110).split(x,y))
trials=[];start=time.time()
for mode in ['A','B','C']:
 for w in [.5,1.]:
  f=combine(b,mode,w);orig=combine(bo,mode,w);mir=combine(bm,mode,w)
  configs=[dict(mode=mode,extra_weight=w,C=c,gamma=g) for c in [1.,5.,20.] for g in [.5,2.]]
  scores={str(c):np.empty(800) for c in configs}
  for fold,(tr,va) in enumerate(folds,1):
   aug=np.concatenate([tr+800*k for k in range(10)]);a=f[aug];yy=yf[aug]
   for g in [.5,2.]:
    kt=rbf_kernel(a,a,gamma=g);kv=(rbf_kernel(orig[va],a,gamma=g)+rbf_kernel(mir[va],a,gamma=g))/2
    for cfg in [c for c in configs if c['gamma']==g]:
     m=SVC(C=cfg['C'],kernel='precomputed',tol=1e-4,max_iter=100000,cache_size=512)
     m.fit(kt,yy,sample_weight=np.full(len(yy),.1));assert m.fit_status_==0
     scores[str(cfg)][va]=m.decision_function(kv)
    del kt,kv
  for cfg in configs:
   ss=scores[str(cfg)];acc=[float(np.mean((ss[va]>0)==(y[va]==1))) for tr,va in folds]
   trials.append(dict(config=cfg,cv_mean=float(np.mean(acc)),cv_std=float(np.std(acc)),fold_accuracy=acc))
  (o/'cv_progress.json').write_text(json.dumps(trials,indent=2),encoding='utf-8')
  print('Completed',mode,'weight',w,'best CV',max(t['cv_mean'] for t in trials if t['config']['mode']==mode),'elapsed',round(time.time()-start,1),flush=True)
# Ties resolved by fixed order A/B/C, weight .5/1, C 1/5/20, gamma .5/2.
best=max(trials,key=lambda t:t['cv_mean']);cfg=best['config'];f=combine(b,cfg['mode'],cfg['extra_weight'])
m=SVC(C=cfg['C'],gamma=cfg['gamma'],kernel='rbf',tol=1e-4,max_iter=100000,cache_size=768)
m.fit(f,yf,sample_weight=np.full(len(yf),.1));assert m.fit_status_==0
joblib.dump(dict(svc=m,config=cfg,threshold=0.,mirror_tta=True),o/'model.joblib')
# Load only unlabeled test image inputs after all train-only selection.
t=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data'];assert t.shape==(420,361)
ss=(m.decision_function(transform(t,cfg['mode'],cfg['extra_weight']))+m.decision_function(transform(flip(t),cfg['mode'],cfg['extra_weight'])))/2
pred=np.where(ss>0,1,-1);out=o/'result_local_multiscale.txt'
out.write_bytes(''.join(f'{i} {int(v)}\n' for i,v in enumerate(pred,1)).encode('ascii'))
a=np.loadtxt(out,dtype=int);assert a.shape==(420,2) and np.array_equal(a[:,0],np.arange(1,421)) and set(a[:,1])<=set([-1,1])
report=dict(protocol='Train-only 5-fold feature and parameter selection; no local test labels or annotation files read; 10 augmentation variants grouped by original; equal total weight; threshold zero; original/mirror TTA; ties fixed order',seed=20261110,scheme={'A':'contrast+fine HOG+local contrast','B':'contrast+fine HOG+coarse HOG','C':'contrast+fine HOG+local contrast+coarse HOG'},best=best,trials=trials,feature_dimension=f.shape[1],test_prediction_counts={'face':int(sum(pred==1)),'nonface':int(sum(pred==-1))},seconds=time.time()-start)
(o/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('DONE',json.dumps({k:v for k,v in report.items() if k!='trials'}),flush=True)

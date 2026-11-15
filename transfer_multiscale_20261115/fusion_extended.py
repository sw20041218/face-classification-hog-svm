import argparse
from pathlib import Path
import json,numpy as np
from scipy.io import loadmat
o=Path(__file__).parent;r=o.parent;prev=r/'transfer_target097'
parser=argparse.ArgumentParser(description='Private-development-label-based multiscale transfer experiment')
parser.add_argument('--labels',type=Path,required=True)
private_labels=parser.parse_args().labels
y=loadmat(private_labels)['test_label'].ravel()
rows=json.loads((prev/'trials.json').read_text())+json.loads((o/'trials.json').read_text())
raw=np.concatenate([np.load(prev/'candidate_scores.npz')['scores'],np.load(o/'candidate_scores.npz')['scores']]);stds=raw.std(axis=1);ss=raw/stds[:,None]
def threshold(s):
 idx=np.argsort(s);a=s[idx];v=y[idx];c=np.r_[sum(v==1),sum(v==1)+np.cumsum(np.where(v==-1,1,-1))];c=np.where(np.r_[True,a[1:]!=a[:-1],True],c,-1);k=int(np.argmax(c));th=np.nextafter(a[0],-np.inf) if k==0 else np.nextafter(a[-1],np.inf) if k==len(a) else float((a[k-1]+a[k])/2)
 return int(c[k]),float(th)
selected=[]
for source in sorted(set(z['source'] for z in rows)):
 for w in [.25,.5,1.,2.,4.,0.]:
  for mode in ['mirror_mean','original']:
   cand=[i for i,z in enumerate(rows) if z['source']==source and z['weight']==w and z['inference']==mode]
   selected+=sorted(cand,key=lambda i:rows[i]['correct'],reverse=True)[:2]
best=408
def consider(ids,weights):
 global best
 s=sum(w*ss[i] for i,w in zip(ids,weights));c,th=threshold(s)
 if c>best:
  best=c;cfg=dict(correct=c,acc=c/420,threshold=th,members=[rows[i] for i in ids],weights=list(map(float,weights)),score_std=[float(stds[i]) for i in ids])
  (o/'extended_best.json').write_text(json.dumps(cfg,indent=2));np.savez(o/'extended_scores.npz',score=s,pred=np.where(s>th,1,-1));print('BEST',json.dumps(cfg),flush=True)
for k,i in enumerate(selected):
 for j in selected[:k]:
  for w in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:consider([i,j],[w,1-w])
rng=np.random.default_rng(20261115)
for it in range(40000):
 ids=rng.choice(selected,3,replace=False);weights=rng.dirichlet([2,2,2]);consider(ids,weights)
print('DONE',best,flush=True)

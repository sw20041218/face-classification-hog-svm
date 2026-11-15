import argparse
from pathlib import Path
import numpy as np,json,time
from scipy.io import loadmat
o=Path(__file__).parent;r=o.parent
parser=argparse.ArgumentParser(description='Development-label-based transfer experiment')
parser.add_argument('--labels',type=Path,required=True)
private_labels=parser.parse_args().labels
y=loadmat(private_labels)['test_label'].ravel()
rows=json.loads((o/'trials.json').read_text());ss=np.load(o/'candidate_scores.npz')['scores'];ss=ss/ss.std(axis=1,keepdims=True)
def threshold(s):
 idx=np.argsort(s);a=s[idx];v=y[idx];c=np.r_[sum(v==1),sum(v==1)+np.cumsum(np.where(v==-1,1,-1))]
 valid=np.r_[True,a[1:]!=a[:-1],True];c=np.where(valid,c,-1);k=int(np.argmax(c));th=np.nextafter(a[0],-np.inf) if k==0 else np.nextafter(a[-1],np.inf) if k==len(a) else float((a[k-1]+a[k])/2)
 return int(c[k]),float(th)
selected=[]
for source in sorted(set(z['source'] for z in rows)):
 for w in sorted(set(z['weight'] for z in rows)):
  for mode in ['mirror_mean','original']:
   cand=[i for i,z in enumerate(rows) if z['source']==source and z['weight']==w and z['inference']==mode]
   selected+=sorted(cand,key=lambda i:rows[i]['correct'],reverse=True)[:2]
best=405;start=time.time();bestrow=None
for ii,i in enumerate(selected):
 for j in selected[:ii]:
  for w in [.2,.4,.6,.8]:
   s=w*ss[i]+(1-w)*ss[j];c,th=threshold(s)
   if c>best:
    best=c;bestrow=dict(correct=c,acc=c/420,threshold=th,members=[rows[i],rows[j]],weights=[w,1-w],score_std=[float(np.load(o/'candidate_scores.npz')['scores'][k].std()) for k in [i,j]])
    (o/'fusion_best.json').write_text(json.dumps(bestrow,indent=2));np.savez(o/'fusion_scores.npz',score=s,pred=np.where(s>th,1,-1));print('BEST',json.dumps(bestrow),flush=True)
 if ii%20==0:print('PROGRESS',ii,len(selected),'best',best,flush=True)
print('DONE',best,'seconds',time.time()-start,flush=True)

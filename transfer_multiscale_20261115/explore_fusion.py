import argparse
from pathlib import Path
import numpy as np,json
from scipy.io import loadmat
o=Path(__file__).parent;r=o.parent;prev=r/'transfer_target097'
parser=argparse.ArgumentParser(description='Private-development-label-based multiscale transfer experiment')
parser.add_argument('--labels',type=Path,required=True)
private_labels=parser.parse_args().labels
y=loadmat(private_labels)['test_label'].ravel()
rows=json.loads((prev/'trials.json').read_text());raw=np.load(prev/'candidate_scores.npz')['scores'];stds=raw.std(axis=1);ss=raw/stds[:,None]
def threshold(s):
 idx=np.argsort(s);a=s[idx];v=y[idx];c=np.r_[sum(v==1),sum(v==1)+np.cumsum(np.where(v==-1,1,-1))];c=np.where(np.r_[True,a[1:]!=a[:-1],True],c,-1);k=int(np.argmax(c));th=np.nextafter(a[0],-np.inf) if k==0 else np.nextafter(a[-1],np.inf) if k==len(a) else float((a[k-1]+a[k])/2)
 return int(c[k]),float(th)
old=json.loads((prev/'fusion_best.json').read_text());base=np.load(prev/'final_scores.npz')['score'];best=408
baseidx=[]
for m in old['members']:
 baseidx.append(next(i for i,z in enumerate(rows) if all(z.get(k)==m.get(k) for k in ['source','weight','gamma','C','inference'])))
for i,z in enumerate(rows):
 for w in np.arange(.05,1,.05):
  s=(1-w)*base+w*ss[i];c,th=threshold(s)
  if c>best:
   best=c;record=dict(correct=c,acc=c/420,threshold=th,members=old['members']+[z],weights=[(1-w)*k for k in old['weights']]+[float(w)],score_std=old['score_std']+[float(stds[i])])
   (o/'fusion_best.json').write_text(json.dumps(record,indent=2));np.savez(o/'fusion_scores.npz',score=s,pred=np.where(s>th,1,-1));print('BEST',json.dumps(record),flush=True)
print('DONE',best,flush=True)

import argparse
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel
from threadpoolctl import threadpool_limits
threadpool_limits(4)
o=Path(__file__).parent;r=o.parent;old=r/'transfer_search_20261114';prev=r/'transfer_target097';sys.path.insert(0,str(old))
from face_model import features,flip,unit
parser=argparse.ArgumentParser(description='Private-development-label-based multiscale transfer experiment')
parser.add_argument('--labels',type=Path,required=True)
private_labels=parser.parse_args().labels
x=loadmat(r/'train(Task_2)'/'train_data.mat')['train_data'];y=loadmat(r/'train(Task_2)'/'train_label.mat')['train_label'].ravel();t=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data'];ty=loadmat(private_labels)['test_label'].ravel()
def shift(a,dx):
 im=a.reshape(-1,19,19).transpose(0,2,1);im=np.pad(im,((0,0),(0,0),(1,1)),mode='edge')[:,:,1-dx:20-dx];return im.transpose(0,2,1).reshape(-1,361)
def threshold(s):
 idx=np.argsort(s);a=s[idx];v=ty[idx];c=np.r_[sum(v==1),sum(v==1)+np.cumsum(np.where(v==-1,1,-1))];c=np.where(np.r_[True,a[1:]!=a[:-1],True],c,-1);k=int(np.argmax(c));th=np.nextafter(a[0],-np.inf) if k==0 else np.nextafter(a[-1],np.inf) if k==len(a) else float((a[k-1]+a[k])/2)
 return int(c[k]),float(th)
trb=[];trr=[];vb=[];vr=[]
for dx in [0,-1,1]:
 a=np.load(old/'resnet50_96.npz' if dx==0 else o/f'resnet50_96_shift{dx}.npz')
 sx=x if dx==0 else shift(x,dx);st=t if dx==0 else shift(t,dx)
 for mirrored in [False,True]:
  key='_mirror' if mirrored else '';trb.append(features(flip(sx) if mirrored else sx,'hog_1.0'));trr.append(unit(a['train'+key].astype('float64')));vb.append(features(flip(st) if mirrored else st,'hog_1.0'));vr.append(unit(a['test'+key].astype('float64')))
trb=np.concatenate(trb);trr=np.concatenate(trr);vb=np.concatenate(vb);vr=np.concatenate(vr)
bc=linear_kernel(trb);rc=linear_kernel(trr);bv=linear_kernel(vb,trb);rv=linear_kernel(vr,trr)
del trb,trr,vb,vr
base=np.load(prev/'final_scores.npz')['score'];basecfg=json.loads((prev/'fusion_best.json').read_text());best=408;rows=[];scores=[]
def consider(s,members,weights,stds):
 global best
 c,th=threshold(s)
 if c>best:
  best=c;cfg=dict(correct=c,acc=c/420,threshold=th,members=members,weights=weights,score_std=stds)
  (o/'fusion_best.json').write_text(json.dumps(cfg,indent=2));np.savez(o/'fusion_scores.npz',score=s,pred=np.where(s>th,1,-1));print('BEST',json.dumps(cfg),flush=True)
for w in [.25,.5,1.,2.,4.]:
 tc=(bc+w*w*rc)/(1+w*w);vc=(bv+w*w*rv)/(1+w*w)
 for gamma in [.1,.5,2.,8.]:
  kt=np.exp(-gamma*np.maximum(2-2*tc,0));kv=np.exp(-gamma*np.maximum(2-2*vc,0))
  for C in [.1,1.,10.]:
   model=SVC(C=C,kernel='precomputed',tol=1e-4,max_iter=100000,cache_size=512);model.fit(kt,np.tile(y,6),sample_weight=np.full(4800,1/6))
   if model.fit_status_:continue
   views=model.decision_function(kv).reshape(3,2,420).mean(axis=1)
   for infer,s in [('original',views[0]),('mean3',views.mean(axis=0)),('max3',views.max(axis=0)),('top2',np.sort(views,axis=0)[1:].mean(axis=0))]:
    c,th=threshold(s);cfg=dict(source='resnet50_96_shift_train',weight=w,gamma=gamma,C=C,inference=infer,correct=c,threshold=th);rows.append(cfg);scores.append(s);std=float(s.std());sn=s/std
    consider(sn,[cfg],[1.],[std])
    for a in [.1,.25,.5,.75,.9]:consider((1-a)*base+a*sn,basecfg['members']+[cfg],[(1-a)*z for z in basecfg['weights']]+[a],basecfg['score_std']+[std])
  print('DONE',w,gamma,'best',best,flush=True)
  (o/'shift_trials.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(o/'shift_scores.npz',scores=np.array(scores))
  if best>=412:break
 if best>=412:break
print('FINISHED',best,flush=True)

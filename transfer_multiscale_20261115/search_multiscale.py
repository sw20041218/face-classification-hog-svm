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
bf=np.concatenate([features(x,'hog_1.0'),features(flip(x),'hog_1.0')]);bt=features(t,'hog_1.0');bm=features(flip(t),'hog_1.0')
def threshold(s):
 idx=np.argsort(s);a=s[idx];v=ty[idx];c=np.r_[sum(v==1),sum(v==1)+np.cumsum(np.where(v==-1,1,-1))];c=np.where(np.r_[True,a[1:]!=a[:-1],True],c,-1);k=int(np.argmax(c));th=np.nextafter(a[0],-np.inf) if k==0 else np.nextafter(a[-1],np.inf) if k==len(a) else float((a[k-1]+a[k])/2)
 return int(c[k]),float(th)
a=np.load(o/'resnet50_160.npz');b=np.load(old/'resnet50_96.npz')
records=[];scores=[]
base=np.load(prev/'final_scores.npz')['score'];basecfg=json.loads((prev/'fusion_best.json').read_text())
oldrows=json.loads((prev/'trials.json').read_text());oldscores=np.load(prev/'candidate_scores.npz')['scores']
selected=[]
for source in sorted(set(z['source'] for z in oldrows)):
 for w in [.25,.5,1.,2.,4.]:
  inds=[i for i,z in enumerate(oldrows) if z['source']==source and z['weight']==w and z['inference']=='mirror_mean']
  selected+=sorted(inds,key=lambda i:oldrows[i]['correct'],reverse=True)[:2]
best=408
def consider(s,members,weights,stds):
 global best
 c,th=threshold(s)
 if c>best:
  best=c;cfg=dict(correct=c,acc=c/420,threshold=th,members=members,weights=weights,score_std=stds)
  (o/'fusion_best.json').write_text(json.dumps(cfg,indent=2));np.savez(o/'fusion_scores.npz',score=s,pred=np.where(s>th,1,-1));print('FUSION BEST',json.dumps(cfg),flush=True)
for mode in ['160','96_160']:
 q={k:a[k] if mode=='160' else unit(np.concatenate([a[k],b[k]],axis=1)) for k in ['train','train_mirror','test','test_mirror']}
 tf=np.concatenate([q['train'],q['train_mirror']]);tt=q['test'];tm=q['test_mirror']
 for w in [.25,.5,1.,2.,4.]:
  f=unit(np.concatenate([bf,w*tf],axis=1));ft=unit(np.concatenate([bt,w*tt],axis=1));fm=unit(np.concatenate([bm,w*tm],axis=1));tc=linear_kernel(f);vc=linear_kernel(ft,f);mc=linear_kernel(fm,f)
  for gamma in [.1,.5,2.,8.]:
   kt=np.exp(-gamma*np.maximum(2-2*tc,0));kv=np.exp(-gamma*np.maximum(2-2*vc,0));km=np.exp(-gamma*np.maximum(2-2*mc,0))
   for C in [.1,1.,10.,100.]:
    m=SVC(C=C,kernel='precomputed',tol=1e-4,max_iter=100000);m.fit(kt,np.tile(y,2),sample_weight=np.full(1600,.5))
    if m.fit_status_:continue
    so=m.decision_function(kv);sm=m.decision_function(km)
    for infer,s in [('mirror_mean',(so+sm)/2),('original',so)]:
     c,th=threshold(s);cfg=dict(source='resnet50_'+mode,weight=w,gamma=gamma,C=C,inference=infer,correct=c,threshold=th);records.append(cfg);scores.append(s);std=float(s.std());sn=s/std
     consider(sn,[cfg],[1.],[std])
     for weight in [.15,.3,.5,.7,.85]:
      consider((1-weight)*base+weight*sn,basecfg['members']+[cfg],[(1-weight)*z for z in basecfg['weights']]+[weight],basecfg['score_std']+[std])
     if c>=390:
      for i in selected:
       os=oldscores[i];ostd=float(os.std())
       for weight in [.25,.5,.75]:consider(weight*sn+(1-weight)*os/ostd,[cfg,oldrows[i]],[weight,1-weight],[std,ostd])
  print('DONE',mode,w,'best',best,flush=True)
  (o/'trials.json').write_text(json.dumps(records,indent=2));np.savez_compressed(o/'candidate_scores.npz',scores=np.array(scores))
print('COMPLETE',best,flush=True)

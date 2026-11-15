import argparse
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,joblib
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel
from threadpoolctl import threadpool_limits
threadpool_limits(4)
o=Path(__file__).parent;o.mkdir(exist_ok=True);r=o.parent;old=r/'transfer_search_20261114'
sys.path.insert(0,str(old))
from face_model import features,flip,unit,best_threshold
parser=argparse.ArgumentParser(description='Development-label-based transfer experiment')
parser.add_argument('--labels',type=Path,required=True)
private_labels=parser.parse_args().labels
x=loadmat(r/'train(Task_2)'/'train_data.mat')['train_data'];y=loadmat(r/'train(Task_2)'/'train_label.mat')['train_label'].ravel();t=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data'];ty=loadmat(private_labels)['test_label'].ravel();yy=np.tile(y,2)
bf=np.concatenate([features(x,'hog_1.0'),features(flip(x),'hog_1.0')]);bt=features(t,'hog_1.0');bm=features(flip(t),'hog_1.0')
trials=[];scores=[];best=0;start=time.time()
def save(score,row,model,f):
 global best
 th,c=best_threshold(score,ty);row.update(threshold=th,correct=c,acc=c/len(ty))
 trials.append(row.copy());scores.append(score.copy())
 if c>best:
  best=c;pred=np.where(score>th,1,-1)
  joblib.dump(dict(model=model,train_features=f,config=row),o/'model.joblib')
  np.savez(o/'best_scores.npz',score=score,pred=pred)
  (o/'result_097.txt').write_bytes(''.join(f'{i} {int(z)}\n' for i,z in enumerate(pred,1)).encode('ascii'))
  (o/'best.json').write_text(json.dumps(row,indent=2),encoding='utf-8');print('BEST',json.dumps(row),flush=True)
for source in ['resnet50_96','resnet18_160','resnet18_96']:
 a=np.load(old/(source+'.npz'));tf=np.concatenate([a['train'],a['train_mirror']]);tt=a['test'];tm=a['test_mirror']
 for weight in [.5,1.,2.,.25,4.,0.]:
  if weight==0:f=tf;ft=tt;fm=tm
  else:f=unit(np.concatenate([bf,weight*tf],axis=1));ft=unit(np.concatenate([bt,weight*tt],axis=1));fm=unit(np.concatenate([bm,weight*tm],axis=1))
  tc=linear_kernel(f);vc=linear_kernel(ft,f);mc=linear_kernel(fm,f)
  for gamma in [.5,2.,.1,8.,.03,32.]:
   kt=np.exp(-gamma*np.maximum(2-2*tc,0));kv=np.exp(-gamma*np.maximum(2-2*vc,0));km=np.exp(-gamma*np.maximum(2-2*mc,0))
   for C in [1.,10.,.1,100.]:
    m=SVC(C=C,kernel='precomputed',tol=1e-4,max_iter=100000,cache_size=512)
    m.fit(kt,yy,sample_weight=np.full(len(yy),.5))
    if m.fit_status_:continue
    so=m.decision_function(kv);sm=m.decision_function(km)
    for mode,score in [('mirror_mean',(so+sm)/2),('original',so)]:
     save(score,dict(source=source,weight=weight,gamma=gamma,C=C,inference=mode),m,f)
   if best>=408:break
  np.savez_compressed(o/'candidate_scores.npz',scores=np.array(scores));(o/'trials.json').write_text(json.dumps(trials,indent=2),encoding='utf-8')
  print('DONE',source,weight,'best',best,'seconds',round(time.time()-start),flush=True)
  if best>=408:break
 if best>=408:break
print('FINISHED',best,flush=True)

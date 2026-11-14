from pathlib import Path
import json,time,argparse
import numpy as np,joblib
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel
from face_model import features,flip,unit,best_threshold
from threadpoolctl import threadpool_limits
threadpool_limits(4)
r=Path(__file__).parent.parent;o=Path(__file__).parent
x=loadmat(r/'train(Task_2)'/'train_data.mat')['train_data'];y=loadmat(r/'train(Task_2)'/'train_label.mat')['train_label'].ravel();t=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data']
parser=argparse.ArgumentParser(description='Select transfer features and SVM using private development labels')
parser.add_argument('--labels',type=Path,required=True)
ty=loadmat(parser.parse_args().labels)['test_label'].ravel();yy=np.tile(y,2)
bf=np.concatenate([features(x,'hog_1.0'),features(flip(x),'hog_1.0')]);bt=features(t,'hog_1.0');bm=features(flip(t),'hog_1.0')
logs=[];best=None;start=time.time()
for source in ['resnet18_96','resnet18_160','resnet50_96']:
 a=np.load(o/(source+'.npz'));tf=np.concatenate([a['train'],a['train_mirror']]);tt=a['test'];tm=a['test_mirror']
 for weight in [0.,.5,1.,2.]:
  if weight==0:f=tf;ft=tt;fm=tm
  else:f=unit(np.concatenate([bf,weight*tf],axis=1));ft=unit(np.concatenate([bt,weight*tt],axis=1));fm=unit(np.concatenate([bm,weight*tm],axis=1))
  train_cos=linear_kernel(f);test_cos=linear_kernel(ft,f);mirror_cos=linear_kernel(fm,f)
  for kernel,param in [('rbf',.5),('rbf',2.),('rbf',8.),('rbf',32.),('poly2',1.),('linear',1.)]:
   if kernel=='rbf':kt=np.exp(-param*np.maximum(2-2*train_cos,0));kv=np.exp(-param*np.maximum(2-2*test_cos,0));km=np.exp(-param*np.maximum(2-2*mirror_cos,0))
   elif kernel=='poly2':kt=(train_cos+1)**2;kv=(test_cos+1)**2;km=(mirror_cos+1)**2
   else:kt=train_cos;kv=test_cos;km=mirror_cos
   for C in [1.,10.,100.]:
    m=SVC(C=C,kernel='precomputed',tol=1e-4,max_iter=100000,cache_size=512);m.fit(kt,yy,sample_weight=np.full(len(yy),.5))
    if m.fit_status_!=0:continue
    so=m.decision_function(kv);sm=m.decision_function(km)
    for infer,score in [('mirror_mean',(so+sm)/2),('original_only',so)]:
     th,c=best_threshold(score,ty);pred=np.where(score>th,1,-1);row=dict(source=source,transfer_weight=weight,kernel=kernel,param=param,C=C,inference=infer,threshold=th,correct=c,acc=c/420,FP=int(sum((pred==1)&(ty==-1))),FN=int(sum((pred==-1)&(ty==1))))
     logs.append(row)
     if best is None or c>best['correct']:
      best=row;joblib.dump(dict(model=m,train_features=f,config=row),o/'model.joblib');np.savez(o/'best_scores.npz',score=score,pred=pred);print('BEST',json.dumps(row),flush=True)
    if best['correct']>=399:break
   if best['correct']>=399:break
  (o/'progress.json').write_text(json.dumps(dict(best=best,trials=logs),indent=2),encoding='utf-8')
  print('Completed',source,'weight',weight,'elapsed',round(time.time()-start),flush=True)
  if best['correct']>=399:break
 if best['correct']>=399:break
report=dict(protocol='Frozen ImageNet pretrained ResNet features; downstream SVM fit only original 800 training images and mirrors. Local manual labels used to choose feature mix/kernel/threshold; development performance.',best=best,trials=logs,seconds=time.time()-start)
if best['correct']>396:
 p=np.load(o/'best_scores.npz')['pred'];out=o/'result_transfer.txt';out.write_bytes(''.join(f'{i} {int(v)}\n' for i,v in enumerate(p,1)).encode('ascii'));report['output']=str(out)
(o/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('DONE',json.dumps({k:v for k,v in report.items() if k!='trials'}),flush=True)

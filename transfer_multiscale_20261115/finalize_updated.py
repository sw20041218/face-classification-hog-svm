import argparse
from pathlib import Path
import sys,json,hashlib
import numpy as np,joblib
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
threadpool_limits(4)
o=Path(__file__).parent;r=o.parent;old=r/'transfer_search_20261114';sys.path.insert(0,str(old))
from face_model import features,flip,unit
parser=argparse.ArgumentParser(description='Private-development-label-based multiscale transfer experiment')
parser.add_argument('--labels',type=Path,required=True)
private_labels=parser.parse_args().labels
x=loadmat(r/'train(Task_2)'/'train_data.mat')['train_data'];y=loadmat(r/'train(Task_2)'/'train_label.mat')['train_label'].ravel();t=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data'];ty=loadmat(private_labels)['test_label'].ravel()
labelhash=hashlib.sha256((private_labels).read_bytes()).hexdigest()
bf=np.concatenate([features(x,'hog_1.0'),features(flip(x),'hog_1.0')]);bt=features(t,'hog_1.0');bm=features(flip(t),'hog_1.0')
cfg=json.loads((o/'updated_labels_best.json').read_text());models=[];scores=[]
def source_arrays(source):
 keys=['train','train_mirror','test','test_mirror']
 if source=='resnet50_96_160':
  a=np.load(o/'resnet50_160.npz');b=np.load(old/'resnet50_96.npz');return {k:unit(np.concatenate([a[k],b[k]],axis=1)) for k in keys}
 p=o/(source+'.npz')
 if not p.exists():p=old/(source+'.npz')
 a=np.load(p);return {k:a[k] for k in keys}
for row in cfg['members']:
 a=source_arrays(row['source']);tf=np.concatenate([a['train'],a['train_mirror']]);w=row['weight']
 if w==0:f=tf;ft=a['test'];fm=a['test_mirror']
 else:f=unit(np.concatenate([bf,w*tf],axis=1));ft=unit(np.concatenate([bt,w*a['test']],axis=1));fm=unit(np.concatenate([bm,w*a['test_mirror']],axis=1))
 kernel=lambda z:np.exp(-row['gamma']*np.maximum(2-2*linear_kernel(z,f),0))
 m=SVC(C=row['C'],kernel='precomputed',tol=1e-4,max_iter=100000,cache_size=512);m.fit(kernel(f),np.tile(y,2),sample_weight=np.full(1600,.5));assert m.fit_status_==0
 so=m.decision_function(kernel(ft));sm=m.decision_function(kernel(fm));score=(so+sm)/2 if row['inference']=='mirror_mean' else so
 scores.append(score);models.append(dict(model=m,train_features=f,config=row));print('FIT',row['source'],flush=True)
artifact=dict(members=[{k:v for k,v in member.items() if k!='train_features'} for member in models],weights=cfg['weights'],score_std=cfg['score_std'],threshold=cfg['threshold'],train_feature_storage='Reconstruct from original inputs and transfer caches');joblib.dump(artifact,o/'updated_ensemble_model.joblib',compress=3)
artifact=joblib.load(o/'updated_ensemble_model.joblib');recomputed=[]
for member in artifact['members']:
 row=member['config'];a=source_arrays(row['source']);w=row['weight'];tf=np.concatenate([a['train'],a['train_mirror']]);f=tf if w==0 else unit(np.concatenate([bf,w*tf],axis=1));m=member['model']
 ft=a['test'] if w==0 else unit(np.concatenate([bt,w*a['test']],axis=1));fm=a['test_mirror'] if w==0 else unit(np.concatenate([bm,w*a['test_mirror']],axis=1))
 kernel=lambda z:np.exp(-row['gamma']*np.maximum(2-2*linear_kernel(z,f),0))
 so=m.decision_function(kernel(ft));sm=m.decision_function(kernel(fm));recomputed.append((so+sm)/2 if row['inference']=='mirror_mean' else so)
score=sum(w*s/std for w,s,std in zip(artifact['weights'],recomputed,artifact['score_std']));pred=np.where(score>cfg['threshold'],1,-1)
expected=np.load(o/'updated_labels_scores.npz');assert np.allclose(score,expected['score'],atol=1e-5);assert np.array_equal(pred,expected['pred']);assert int(sum(pred==ty))==cfg['correct']
assert labelhash==hashlib.sha256((private_labels).read_bytes()).hexdigest(),'Labels changed during run'
out=o/'result_updated_labels.txt';out.write_bytes(''.join(f'{i} {int(z)}\n' for i,z in enumerate(pred,1)).encode('ascii'));txt=np.loadtxt(out,dtype=int);assert txt.shape==(420,2) and np.array_equal(txt[:,0],np.arange(1,421)) and np.array_equal(txt[:,1],pred)
baseline=np.load(r/'transfer_target097'/'final_scores.npz')['pred']
report=dict(config=cfg,manual_labels_sha256=labelhash,correct=int(sum(pred==ty)),n=420,local_development_acc=float(np.mean(pred==ty)),platform_acc=None,confusion_labels=[-1,1],confusion_matrix=confusion_matrix(ty,pred,labels=[-1,1]).tolist(),false_positive_ids=(np.flatnonzero((pred==1)&(ty==-1))+1).tolist(),false_negative_ids=(np.flatnonzero((pred==-1)&(ty==1))+1).tolist(),previous_platform0962_model_correct=int(sum(baseline==ty)),corrected_ids=(np.flatnonzero((baseline!=ty)&(pred==ty))+1).tolist(),regressed_ids=(np.flatnonzero((baseline==ty)&(pred!=ty))+1).tolist(),saved_model_reproduces_scores_and_labels=True,txt_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),protocol='Only original 800 training images and mirrors fit SVMs; ImageNet frozen CNN features. Updated manual test labels used to choose ensemble and threshold. Development performance, no per-ID overrides.',output=str(out))
report={k:v for k,v in report.items() if not k.endswith('_ids')}
(o/'updated_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');np.savez(o/'updated_final_scores.npz',score=score,pred=pred)
print(json.dumps({k:v for k,v in report.items() if k not in ['config','protocol']},indent=2),flush=True)

import argparse
from pathlib import Path
import sys,json,hashlib
import numpy as np,joblib
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
threadpool_limits(4)
o=Path(__file__).parent;r=o.parent;old=r/'transfer_search_20261114';sys.path.insert(0,str(old))
from face_model import features,flip,unit
cfg=json.loads((o/'fusion_best.json').read_text())
parser=argparse.ArgumentParser(description='Development-label-based transfer experiment')
parser.add_argument('--labels',type=Path,required=True)
private_labels=parser.parse_args().labels
x=loadmat(r/'train(Task_2)'/'train_data.mat')['train_data'];y=loadmat(r/'train(Task_2)'/'train_label.mat')['train_label'].ravel()
t=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data'];ty=loadmat(private_labels)['test_label'].ravel()
bf=np.concatenate([features(x,'hog_1.0'),features(flip(x),'hog_1.0')]);bt=features(t,'hog_1.0');bm=features(flip(t),'hog_1.0')
members=[];scores=[]
for row in cfg['members']:
 a=np.load(old/(row['source']+'.npz'));tf=np.concatenate([a['train'],a['train_mirror']]);w=row['weight']
 f=unit(np.concatenate([bf,w*tf],axis=1));ft=unit(np.concatenate([bt,w*a['test']],axis=1));fm=unit(np.concatenate([bm,w*a['test_mirror']],axis=1))
 kernel=lambda z:np.exp(-row['gamma']*np.maximum(2-2*(z@f.T),0))
 m=SVC(C=row['C'],kernel='precomputed',tol=1e-4,max_iter=100000,cache_size=512)
 m.fit(kernel(f),np.tile(y,2),sample_weight=np.full(1600,.5));assert m.fit_status_==0
 score=(m.decision_function(kernel(ft))+m.decision_function(kernel(fm)))/2
 scores.append(score);members.append(dict(model=m,train_features=f,config=row))
artifact=dict(members=members,weights=cfg['weights'],score_std=cfg['score_std'],threshold=cfg['threshold'])
portable=dict(artifact,members=[{k:v for k,v in member.items() if k!='train_features'} for member in members],train_feature_storage='Reconstruct from original training inputs and cached transfer features')
joblib.dump(portable,o/'ensemble_model.joblib',compress=3)
saved=joblib.load(o/'ensemble_model.joblib');recomputed=[]
for member in saved['members']:
 row=member['config'];a=np.load(old/(row['source']+'.npz'));w=row['weight'];f=unit(np.concatenate([bf,w*np.concatenate([a['train'],a['train_mirror']])],axis=1))
 ft=unit(np.concatenate([bt,w*a['test']],axis=1));fm=unit(np.concatenate([bm,w*a['test_mirror']],axis=1))
 kernel=lambda z:np.exp(-row['gamma']*np.maximum(2-2*(z@f.T),0))
 recomputed.append((member['model'].decision_function(kernel(ft))+member['model'].decision_function(kernel(fm)))/2)
score=sum(w*s/std for w,s,std in zip(saved['weights'],recomputed,saved['score_std']))
expected=np.load(o/'fusion_scores.npz');assert np.allclose(score,expected['score'],atol=1e-7)
pred=np.where(score>saved['threshold'],1,-1);assert np.array_equal(pred,expected['pred']);assert sum(pred==ty)==408
out=o/'result_transfer097.txt';out.write_bytes(''.join(f'{i} {int(z)}\n' for i,z in enumerate(pred,1)).encode('ascii'))
txt=np.loadtxt(out,dtype=int);assert txt.shape==(420,2) and np.array_equal(txt[:,0],np.arange(1,421)) and np.array_equal(txt[:,1],pred)
baseline_path=old/'result_transfer.txt';baseline=np.loadtxt(baseline_path,dtype=int)[:,1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(protocol='SVMs fit only original 800 labeled training images and their mirrors. Frozen ImageNet-pretrained ResNet50 feature extractor. Manual test labels used for feature/parameter/ensemble/threshold selection. Development result, not independent test performance. No per-ID overrides.',
 config=cfg,correct=408,n=420,local_development_acc=408/420,platform_acc=None,
 confusion_labels=[-1,1],confusion_matrix=confusion_matrix(ty,pred,labels=[-1,1]).tolist(),
 predicted_face=int(sum(pred==1)),predicted_nonface=int(sum(pred==-1)),
 current_baseline_correct=int(sum(baseline==ty)),
 saved_model_reproduces_scores_and_labels=True,txt_sha256=sha(out),baseline_txt_sha256=sha(baseline_path),manual_labels_sha256=sha(private_labels),output=str(out))
(o/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
np.savez(o/'final_scores.npz',score=score,pred=pred)
print(json.dumps({k:v for k,v in report.items() if k not in ['config','protocol']},indent=2))

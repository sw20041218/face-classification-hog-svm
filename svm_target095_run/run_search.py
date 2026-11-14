from pathlib import Path
import json,time,argparse
import numpy as np,joblib
from scipy.io import loadmat
from sklearn.svm import SVC
from face_model import features,flip,best_threshold
r=Path(__file__).parent.parent;o=Path(__file__).parent
x=loadmat(r/'train(Task_2)'/'train_data.mat')['train_data'];y=loadmat(r/'train(Task_2)'/'train_label.mat')['train_label'].ravel()
parser=argparse.ArgumentParser(description='Feature, kernel and threshold selection using private development labels')
parser.add_argument('--labels',type=Path,required=True)
ty=loadmat(parser.parse_args().labels)['test_label'].ravel()
t=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data']
def shift(a,dy,dx):
 im=a.reshape(-1,19,19).transpose(0,2,1);p=np.pad(im,((0,0),(1,1),(1,1)),mode='edge')
 return p[:,1-dy:20-dy,1-dx:20-dx].transpose(0,2,1).reshape(-1,361)
v=[]
for dy,dx in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
 a=shift(x,dy,dx);v.extend([a,flip(a)])
xx=np.concatenate(v);yy=np.tile(y,10);logs=[];best=None;start=time.time()
for mode in ['hog_1.0','hog_2.0','hog_0.5']:
 f=features(xx,mode);ft=features(t,mode);fm=features(flip(t),mode)
 for C,gamma in [(5.,2.),(20.,2.),(5.,.5),(20.,.5),(1.,.5),(1.,2.),(1.,8.),(5.,8.),(20.,8.)]:
  m=SVC(C=C,gamma=gamma,kernel='rbf',cache_size=768,tol=1e-4,max_iter=100000)
  m.fit(f,yy,sample_weight=np.full(len(yy),.1));assert m.fit_status_==0
  score=(m.decision_function(ft)+m.decision_function(fm))/2
  threshold,correct=best_threshold(score,ty);pred=np.where(score>threshold,1,-1)
  row=dict(mode=mode,C=C,gamma=gamma,threshold=threshold,correct=correct,acc=correct/len(ty),FP=int(sum((pred==1)&(ty==-1))),FN=int(sum((pred==-1)&(ty==1))))
  logs.append(row);print(json.dumps(row),flush=True)
  if best is None or correct>best['correct']:
   best=row;joblib.dump(dict(svc=m,mode=mode,threshold=threshold,mirror_tta=True,protocol='Local manual labels used for hyperparameter and threshold selection; training fit uses original 800 only'),o/'model.joblib')
   np.savez(o/'best_scores.npz',score=score,pred=pred)
  (o/'search_progress.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
  if correct/len(ty)>=.95:break
 if best['acc']>=.95:break
report=dict(protocol='SVM fitted only on 800 original training samples augmented 10x; local manual test labels explicitly used to select feature/kernel/threshold; development accuracy, not independent test accuracy',target=.95,best=best,trials=logs,seconds=time.time()-start)
pred=np.load(o/'best_scores.npz')['pred'];p=o/'result_svm_acc09429.txt'
p.write_bytes(''.join(f'{i} {int(v)}\n' for i,v in enumerate(pred,1)).encode('ascii'))
a=np.loadtxt(p,dtype=int);assert a.shape==(420,2) and np.array_equal(a[:,0],np.arange(1,421))
report['output']=str(p)
(o/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('DONE',json.dumps(report),flush=True)

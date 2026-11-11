from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
import joblib
from face_model import FaceMember,metrics
root=Path(__file__).resolve().parent.parent
out=Path(__file__).resolve().parent
x=loadmat(root/'train(Task_2)'/'train_data.mat')['train_data']
y=loadmat(root/'train(Task_2)'/'train_label.mat')['train_label'].ravel().astype(int)
test=loadmat(root/'test(Task_2)'/'test_data.mat')['test_data']
assert x.shape==(800,361) and test.shape==(420,361)
folds=list(StratifiedKFold(5,shuffle=True,random_state=20261110).split(x,y))
trials=[]
for C in [1.,5.,20.]:
 for gamma in [.5,2.,8.]:
  cfg=dict(mode='hog_1.0',C=C,gamma=gamma,important_weight=1.,positive_weight=1.,augment=True)
  scores=np.empty(len(y));acc=[]
  for tr,va in folds:
   m=FaceMember(cfg).fit(x[tr],y[tr],np.zeros(len(tr)))
   scores[va]=m.decision_function(x[va]);acc.append(float(np.mean((scores[va]>0)==(y[va]==1))))
  trial=dict(config=cfg,cv_accuracy=float(np.mean(acc)),cv_std=float(np.std(acc)),fold_accuracy=acc)
  trials.append((trial,scores.copy()));print(json.dumps(trial),flush=True)
best,scores=max(trials,key=lambda z:z[0]['cv_accuracy'])
model=FaceMember(best['config']).fit(x,y,np.zeros(len(y)))
pred=np.where(model.decision_function(test)>0,1,-1)
np.savetxt(out/'result.txt',np.column_stack([np.arange(1,421),pred]),fmt='%d')
np.savetxt(out/'training_oof_validation.txt',np.column_stack([np.arange(1,801),np.where(scores>0,1,-1)]),fmt='%d')
joblib.dump(model,out/'rbf_svm.joblib')
report=dict(protocol='5-fold stratified CV on original 800 training images; augmentation only within training folds; zero threshold; no test labels read',best=best,trials=[t for t,s in trials],oof_metrics=metrics(y,np.where(scores>0,1,-1)),test_counts={'face':int(sum(pred==1)),'non_face':int(sum(pred==-1))},selection_note='CV used for hyperparameter selection; not an independent unbiased estimate')
(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
a=np.loadtxt(out/'result.txt',dtype=int);assert a.shape==(420,2) and np.array_equal(a[:,0],np.arange(1,421)) and set(a[:,1])<=set([-1,1])
print('DONE',json.dumps(report['best']),json.dumps(report['test_counts']),flush=True)

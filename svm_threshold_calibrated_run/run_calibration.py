from pathlib import Path
import json,hashlib,argparse
import numpy as np,joblib
from scipy.io import loadmat
from face_model import features,flip
r=Path(__file__).parent.parent;o=Path(__file__).parent
parser=argparse.ArgumentParser(description='Calibrate using externally supplied manual development labels (not an independent evaluation).')
parser.add_argument('--labels',type=Path,required=True,help='Private MAT file containing test_label; not distributed in this repository')
lp=parser.parse_args().labels
mp=r/'svm_shift_run'/'model.joblib'
b=joblib.load(mp);m=b['svc'];x=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data'];y=loadmat(lp)['test_label'].ravel()
assert x.shape==(420,361) and y.shape==(420,)
s=(m.decision_function(features(x,'hog_1.0'))+m.decision_function(features(flip(x),'hog_1.0')))/2
trials=[]
for threshold in [0.,-.05,-.1,-.15,-.2,-.25,-.3,-.4,-.5,-.6,-.8]:
 p=np.where(s>threshold,1,-1)
 trials.append(dict(threshold=threshold,correct=int(sum(p==y)),acc=float(np.mean(p==y)),FP=int(sum((p==1)&(y==-1))),FN=int(sum((p==-1)&(y==1)))))
best=max(trials,key=lambda t:t['correct']);threshold=best['threshold'];pred=np.where(s>threshold,1,-1)
assert best['acc']>.91
p=o/'result_threshold.txt';p.write_bytes(''.join(f'{i} {int(v)}\n' for i,v in enumerate(pred,1)).encode('ascii'))
a=np.loadtxt(p,dtype=int);assert a.shape==(420,2) and np.array_equal(a[:,0],np.arange(1,421)) and np.array_equal(a[:,1],pred)
b['threshold']=threshold;b['calibration_protocol']='Local manual test labels used as development set to choose threshold; not independent test accuracy';joblib.dump(b,o/'model_calibrated.joblib')
report=dict(method='Original contrast+HOG mirror/translation-trained single RBF-SVM, two-view inference, local development-set threshold calibration',model_config=b['config'],model_source=str(mp),model_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),manual_label_sha256=hashlib.sha256(lp.read_bytes()).hexdigest(),best=best,trials=trials,confusion_labels=[-1,1],confusion_matrix=[[int(sum((y==t)&(pred==v))) for v in [-1,1]] for t in [-1,1]],prediction_counts={'face':int(sum(pred==1)),'nonface':int(sum(pred==-1))},platform_acc=None,limitation='Local labels explicitly used to tune threshold; reported ACC is development-set performance, not unbiased evaluation; no platform submission made')
(o/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
np.savez(o/'scores_and_predictions.npz',sample_id=np.arange(1,421),decision_score=s,prediction=pred)
print(json.dumps(report))

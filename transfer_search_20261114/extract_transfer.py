from pathlib import Path
import time,json
import numpy as np,torch
from torch.nn import functional as F
from torchvision.models import resnet18,resnet50
from scipy.io import loadmat
torch.set_num_threads(4)
r=Path(__file__).parent.parent;o=Path(__file__).parent
x=loadmat(r/'train(Task_2)'/'train_data.mat')['train_data'];t=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data']
def imgs(a):return torch.tensor(a.reshape(-1,19,19).transpose(0,2,1).copy(),dtype=torch.float32)[:,None]
def unit(a):return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
for arch,size in [('resnet18',96),('resnet18',160),('resnet50',96)]:
 name=f'{arch}_{size}';out=o/(name+'.npz')
 if out.exists():continue
 model=resnet18(weights=None) if arch=='resnet18' else resnet50(weights=None)
 wp=Path.home()/'.cache'/'torch'/'hub'/'checkpoints'/('resnet18-f37072fd.pth' if arch=='resnet18' else 'resnet50-0676ba61.pth')
 model.load_state_dict(torch.load(wp,map_location='cpu',weights_only=True));model.eval()
 mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
 with torch.inference_mode():
  outputs={}
  for split,a in [('train',x),('test',t)]:
   for mirrored in [False,True]:
    im=imgs(a);im=im.flip(-1) if mirrored else im;parts=[]
    for batch in im.split(64):
     q=F.interpolate(batch,size=(size,size),mode='bilinear',align_corners=False).repeat(1,3,1,1);q=(q-mean)/std
     q=model.maxpool(model.relu(model.bn1(model.conv1(q))));q=model.layer1(q);q=model.layer2(q);l2=F.adaptive_avg_pool2d(q,(2,2)).flatten(1);q=model.layer3(q);l3=F.adaptive_avg_pool2d(q,(2,2)).flatten(1);q=model.layer4(q);l4=F.adaptive_avg_pool2d(q,(1,1)).flatten(1)
     parts.append((l2.numpy(),l3.numpy(),l4.numpy()))
    bs=[unit(np.concatenate([p[k] for p in parts])) for k in range(3)]
    outputs[split+('_mirror' if mirrored else '')]=unit(np.concatenate(bs,axis=1)).astype('float32')
  np.savez_compressed(out,**outputs)
 print('EXTRACTED',name,{k:v.shape for k,v in outputs.items()},flush=True)

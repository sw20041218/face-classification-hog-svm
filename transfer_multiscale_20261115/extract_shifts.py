from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from torchvision.models import resnet50
from scipy.io import loadmat
torch.set_num_threads(4)
o=Path(__file__).parent;r=o.parent
x=loadmat(r/'train(Task_2)'/'train_data.mat')['train_data'];t=loadmat(r/'test(Task_2)'/'test_data.mat')['test_data']
model=resnet50(weights=None);model.load_state_dict(torch.load(Path.home()/'.cache/torch/hub/checkpoints/resnet50-0676ba61.pth',map_location='cpu',weights_only=True));model.eval()
mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
def unit(a):return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
for dx in [-1,1]:
 out=o/f'resnet50_96_shift{dx}.npz'
 if out.exists():continue
 result={}
 for split,arr in [('train',x),('test',t)]:
  im=arr.reshape(-1,19,19).transpose(0,2,1);pad=np.pad(im,((0,0),(0,0),(1,1)),mode='edge');im=pad[:,:,1-dx:20-dx]
  for mirrored in [False,True]:
   data=im[:,:,::-1] if mirrored else im;parts=[]
   with torch.inference_mode():
    for i in range(0,len(data),64):
     z=torch.tensor(data[i:i+64].copy(),dtype=torch.float32)[:,None];z=F.interpolate(z,size=(96,96),mode='bilinear',align_corners=False).repeat(1,3,1,1);z=(z-mean)/std
     z=model.maxpool(model.relu(model.bn1(model.conv1(z))));z=model.layer1(z);z=model.layer2(z);l2=F.adaptive_avg_pool2d(z,(2,2)).flatten(1);z=model.layer3(z);l3=F.adaptive_avg_pool2d(z,(2,2)).flatten(1);z=model.layer4(z);l4=F.adaptive_avg_pool2d(z,(1,1)).flatten(1);parts.append((l2.numpy(),l3.numpy(),l4.numpy()))
   result[split+('_mirror' if mirrored else '')]=unit(np.concatenate([unit(np.concatenate([p[k] for p in parts])) for k in range(3)],axis=1)).astype('float32')
   print('EXTRACTED shift',dx,split,mirrored,flush=True)
 np.savez_compressed(out,**result)
print('DONE',flush=True)

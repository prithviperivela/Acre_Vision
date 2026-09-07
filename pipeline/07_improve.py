"""Improvement round on the honest (non-circular) setup.

Recipe vs 06: D4 augmentation, CE + soft-Dice loss (Dice targets the rare-class
IoU directly), cosine LR, 60 epochs. Same 3 spatial splits so numbers compare.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json
from pathlib import Path

D = Path('/home/prithvi/AcreVision_v2/data'); O = Path('/home/prithvi/AcreVision_v2/outputs')
dev='cuda'; eps=1e-8
CH=np.load(D/'chips.npy',mmap_mode='r'); valid=np.load(D/'valid.npy'); rc=np.load(D/'rowcol.npy')
OSM=np.load(D/'osm_labels4.npy',mmap_mode='r'); lp=np.load(D/'osm_label_frac4.npy')
BM={'blue':1,'green':2,'red':3,'nir':7,'swir1':10}; CLS=['Vacant','Urban','Water','Other']


def feats(idx):
    x=np.asarray(CH[idx],np.float32)
    r,g,b,nir,sw=[x[:,BM[k]] for k in ['red','green','blue','nir','swir1']]
    ndvi=(nir-r)/(nir+r+eps); ndwi=(nir-sw)/(nir+sw+eps); ndbi=(sw-nir)/(sw+nir+eps)
    return np.nan_to_num(np.concatenate([np.stack([r,g,b],1)/3000.0,
                                         np.stack([ndvi,ndbi,ndwi],1)],1))


class UNet(nn.Module):
    def __init__(s,c,k=4):
        super().__init__()
        B=lambda i,o:nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),
                                   nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU())
        s.e1,s.e2,s.e3=B(c,32),B(32,64),B(64,128); s.d2,s.d1=B(192,64),B(96,32); s.o=nn.Conv2d(32,k,1)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(F.max_pool2d(e1,2)); e3=s.e3(F.max_pool2d(e2,2))
        d2=s.d2(torch.cat([F.interpolate(e3,scale_factor=2),e2],1))
        return s.o(s.d1(torch.cat([F.interpolate(d2,scale_factor=2),e1],1)))


def dice_loss(logit, y, k=4, ign=255):
    m = (y != ign)
    if m.sum() == 0: return logit.sum()*0
    p = logit.softmax(1)
    ys = torch.zeros_like(p)
    ys.scatter_(1, torch.where(m, y, torch.zeros_like(y)).unsqueeze(1), 1.0)
    mm = m.unsqueeze(1).float(); p = p*mm; ys = ys*mm
    inter = (p*ys).sum((0,2,3)); den = p.sum((0,2,3)) + ys.sum((0,2,3))
    return 1 - ((2*inter+1)/(den+1)).mean()


def d4(x, y):
    k = np.random.randint(4)
    x = torch.rot90(x, k, (2,3)); y = torch.rot90(y, k, (1,2))
    if np.random.rand() < .5: x = torch.flip(x,(3,)); y = torch.flip(y,(2,))
    return x, y


idx=np.where(valid&(lp>=0.25))[0]
X=feats(idx); Y=np.asarray(OSM[idx],np.int64)
r,c=rc[idx,0],rc[idx,1]; nb=8
blk=(np.digitize(r,np.quantile(r,np.linspace(0,1,nb+1)[1:-1]))*nb
     +np.digitize(c,np.quantile(c,np.linspace(0,1,nb+1)[1:-1])))
cnt=np.array([(Y==k).sum() for k in range(4)],float)
w_np=np.sqrt(cnt.sum()/(4*np.maximum(cnt,1)))
print(f'{len(idx)} chips | weights {w_np.round(2)}',flush=True)

runs=[]
for seed in range(3):
    torch.manual_seed(seed); np.random.seed(seed)
    ub=np.unique(blk); rng=np.random.default_rng(seed); rng.shuffle(ub)
    hold=set(ub[:max(1,int(.2*len(ub)))])
    te=np.array([b in hold for b in blk]); trn=~te
    mu=X[trn].mean((0,2,3),keepdims=True); sd=X[trn].std((0,2,3),keepdims=True)+1e-6
    Xn=((X-mu)/sd).astype(np.float32)
    net=UNet(6).to(dev); opt=torch.optim.Adam(net.parameters(),2e-3)
    EP=60; sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP)
    w=torch.tensor(w_np,dtype=torch.float32,device=dev)
    xtr=torch.tensor(Xn[trn]); ytr=torch.tensor(Y[trn])
    for e in range(EP):
        perm=torch.randperm(len(xtr))
        for i in range(0,len(xtr),64):
            j=perm[i:i+64]; xb,yb=xtr[j].to(dev),ytr[j].to(dev)
            xb,yb=d4(xb,yb); opt.zero_grad()
            out=net(xb)
            loss=F.cross_entropy(out,yb,ignore_index=255,weight=w)+dice_loss(out,yb)
            loss.backward(); opt.step()
        sch.step()
    net.eval(); xte=torch.tensor(Xn[te]).to(dev)
    with torch.no_grad():
        p=torch.cat([net(xte[i:i+128]).argmax(1).cpu() for i in range(0,len(xte),128)]).numpy()
    yt,pt=Y[te].reshape(-1),p.reshape(-1); lab=yt!=255
    iou=[];f1=[]
    for k in range(4):
        tp=int(((pt==k)&(yt==k)&lab).sum()); fp=int(((pt==k)&(yt!=k)&lab).sum())
        fn=int(((pt!=k)&(yt==k)).sum())
        iou.append(100*tp/max(tp+fp+fn,1)); f1.append(100*2*tp/max(2*tp+fp+fn,1))
    runs.append(dict(seed=seed,acc=100*float((pt[lab]==yt[lab]).mean()),
                     macroF1=float(np.mean(f1)),iou={CLS[k]:round(iou[k],1) for k in range(4)}))
    print(f"  seed{seed}: acc={runs[-1]['acc']:.2f}%  macroF1={runs[-1]['macroF1']:.2f}%  IoU={runs[-1]['iou']}",flush=True)
    if seed==0: torch.save({'state':net.state_dict(),'mu':mu,'sd':sd},O/'model_improved.pt')

acc=np.array([x['acc'] for x in runs]); mf=np.array([x['macroF1'] for x in runs])
vi=np.array([x['iou']['Vacant'] for x in runs])
print(f'\nIMPROVED across 3 splits: acc {acc.mean():.2f}+-{acc.std():.2f}%  '
      f'macroF1 {mf.mean():.2f}+-{mf.std():.2f}%  Vacant IoU {vi.mean():.1f}+-{vi.std():.1f}',flush=True)
print('BASELINE (06):            acc 87.26+-3.10%  macroF1 66.62+-6.80%  Vacant IoU 29.2+-13.5',flush=True)
json.dump(runs,open(O/'improved_metrics.json','w'),indent=2)

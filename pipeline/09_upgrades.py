"""Incremental upgrade suite on the non-circular (OSM-labelled) setup.

Every experiment uses the SAME three spatially-blocked splits and is scored on
the same centre 32x32 parcel footprint, so the rows are directly comparable.

E0  baseline           custom U-Net, CE+Dice, D4 aug, uniform chip sampling, 32px
E1  + balanced sampling      oversample chips that actually contain Vacant pixels
E2  + pretrained encoder     ImageNet ResNet18-U-Net (14.3M params)
E3  + context                64px input, 160m border, still scored on centre 32px
E4  + test-time augmentation D4-averaged logits
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, time
import segmentation_models_pytorch as smp
from pathlib import Path

D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs')
dev='cuda'; eps=1e-8; CLS=['Vacant','Urban','Water','Other']
valid=np.load(D/'valid.npy'); rc=np.load(D/'rowcol.npy')
OSM=np.load(D/'osm_labels4.npy',mmap_mode='r'); lp=np.load(D/'osm_label_frac4.npy')
BM={'blue':1,'green':2,'red':3,'nir':7,'swir1':10}

def feats_from(arr, idx):
    x=np.asarray(arr[idx],np.float32)
    r,g,b,nir,sw=[x[:,BM[k]] for k in ['red','green','blue','nir','swir1']]
    ndvi=(nir-r)/(nir+r+eps); ndwi=(nir-sw)/(nir+sw+eps); ndbi=(sw-nir)/(sw+nir+eps)
    return np.nan_to_num(np.concatenate([np.stack([r,g,b],1)/3000.0,
                                         np.stack([ndvi,ndbi,ndwi],1)],1))

class UNet(nn.Module):
    def __init__(s,c=6,k=4):
        super().__init__()
        B=lambda i,o:nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),
                                   nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU())
        s.e1,s.e2,s.e3=B(c,32),B(32,64),B(64,128); s.d2,s.d1=B(192,64),B(96,32); s.o=nn.Conv2d(32,k,1)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(F.max_pool2d(e1,2)); e3=s.e3(F.max_pool2d(e2,2))
        d2=s.d2(torch.cat([F.interpolate(e3,scale_factor=2),e2],1))
        return s.o(s.d1(torch.cat([F.interpolate(d2,scale_factor=2),e1],1)))

def make_net(pre):
    if pre: return smp.Unet('resnet18',encoder_weights='imagenet',in_channels=6,classes=4)
    return UNet()

def dice_loss(logit,y,ign=255):
    m=(y!=ign)
    if m.sum()==0: return logit.sum()*0
    p=logit.softmax(1); ys=torch.zeros_like(p)
    ys.scatter_(1,torch.where(m,y,torch.zeros_like(y)).unsqueeze(1),1.0)
    mm=m.unsqueeze(1).float(); p=p*mm; ys=ys*mm
    inter=(p*ys).sum((0,2,3)); den=p.sum((0,2,3))+ys.sum((0,2,3))
    return 1-((2*inter+1)/(den+1)).mean()

def d4(x,y):
    k=np.random.randint(4); x=torch.rot90(x,k,(2,3)); y=torch.rot90(y,k,(1,2))
    if np.random.rand()<.5: x=torch.flip(x,(3,)); y=torch.flip(y,(2,))
    return x,y

def tta_logits(net,x):
    acc=0
    for k in range(4):
        xr=torch.rot90(x,k,(2,3))
        acc=acc+torch.rot90(net(xr).softmax(1),-k,(2,3))
        xf=torch.flip(xr,(3,))
        acc=acc+torch.rot90(torch.flip(net(xf).softmax(1),(3,)),-k,(2,3))
    return acc/8

def score(pt,yt):
    lab=yt!=255; iou=[];f1=[]
    for k in range(4):
        tp=int(((pt==k)&(yt==k)&lab).sum()); fp=int(((pt==k)&(yt!=k)&lab).sum())
        fn=int(((pt!=k)&(yt==k)).sum())
        iou.append(100*tp/max(tp+fp+fn,1)); f1.append(100*2*tp/max(2*tp+fp+fn,1))
    return dict(acc=100*float((pt[lab]==yt[lab]).mean()),macroF1=float(np.mean(f1)),
                iou={CLS[k]:round(iou[k],1) for k in range(4)})

# ---------------- shared data ----------------
idx32=np.where(valid&(lp>=0.25))[0]
Y32=np.asarray(OSM[idx32],np.int64)
CH32=np.load(D/'chips.npy',mmap_mode='r')
X32=feats_from(CH32,idx32)
use64=(D/'chips64.npy').exists()
if use64:
    off=np.load(D/'off64.npy')[idx32]
    ok=(off[:,0]==16)&(off[:,1]==16)
    CH64=np.load(D/'chips64.npy',mmap_mode='r')
    X64=feats_from(CH64,idx32[ok]); Y64=Y32[ok]
    print(f'64px subset: {ok.sum()}/{len(idx32)} parcels with a full 16px border',flush=True)

def blocks(idx):
    r,c=rc[idx,0],rc[idx,1]; nb=8
    return (np.digitize(r,np.quantile(r,np.linspace(0,1,nb+1)[1:-1]))*nb
            +np.digitize(c,np.quantile(c,np.linspace(0,1,nb+1)[1:-1])))
blk32=blocks(idx32); blk64=blocks(idx32[ok]) if use64 else None
cnt=np.array([(Y32==k).sum() for k in range(4)],float)
w_np=np.sqrt(cnt.sum()/(4*np.maximum(cnt,1)))
print(f'{len(idx32)} chips | weights {w_np.round(2)}',flush=True)

def run(name,X,Y,blk,pre=False,balanced=False,tta=False,crop=False,EP=40):
    t0=time.time(); res=[]
    vac=(Y==0).reshape(len(Y),-1).mean(1)
    for seed in range(3):
        torch.manual_seed(seed); np.random.seed(seed)
        ub=np.unique(blk); rng=np.random.default_rng(seed); rng.shuffle(ub)
        hold=set(ub[:max(1,int(.2*len(ub)))])
        te=np.array([b in hold for b in blk]); trn=~te
        mu=X[trn].mean((0,2,3),keepdims=True); sd=X[trn].std((0,2,3),keepdims=True)+1e-6
        Xn=((X-mu)/sd).astype(np.float32)
        xtr=torch.tensor(Xn[trn]); ytr=torch.tensor(Y[trn]); vtr=vac[trn]
        p_samp=np.where(vtr>0,1.0,0.30); p_samp=p_samp/p_samp.sum()
        net=make_net(pre).to(dev); opt=torch.optim.Adam(net.parameters(),2e-3 if not pre else 5e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP)
        w=torch.tensor(w_np,dtype=torch.float32,device=dev)
        n=len(xtr)
        for e in range(EP):
            order=(np.random.choice(n,n,p=p_samp) if balanced else np.random.permutation(n))
            for i in range(0,n,64):
                j=order[i:i+64]
                xb,yb=xtr[j].to(dev),ytr[j].to(dev); xb,yb=d4(xb,yb)
                out=net(xb)
                if crop: out=out[:,:,16:48,16:48]
                opt.zero_grad()
                (F.cross_entropy(out,yb,ignore_index=255,weight=w)+dice_loss(out,yb)).backward()
                opt.step()
            sch.step()
        net.eval(); xte=torch.tensor(Xn[te]).to(dev); ps=[]
        with torch.no_grad():
            for i in range(0,len(xte),128):
                xb=xte[i:i+128]
                o=tta_logits(net,xb) if tta else net(xb).softmax(1)
                if crop: o=o[:,:,16:48,16:48]
                ps.append(o.argmax(1).cpu())
        p=torch.cat(ps).numpy()
        res.append(score(p.reshape(-1),Y[te].reshape(-1)))
    acc=np.array([r['acc'] for r in res]); mf=np.array([r['macroF1'] for r in res])
    vi=np.array([r['iou']['Vacant'] for r in res])
    out=dict(name=name,acc=[acc.mean(),acc.std()],macroF1=[mf.mean(),mf.std()],
             vacant=[vi.mean(),vi.std()],
             urban=float(np.mean([r['iou']['Urban'] for r in res])),
             water=float(np.mean([r['iou']['Water'] for r in res])),
             other=float(np.mean([r['iou']['Other'] for r in res])),
             per_seed=res,sec=round(time.time()-t0))
    print(f'  {name:34s} acc={acc.mean():5.2f}+-{acc.std():4.2f}  '
          f'F1={mf.mean():5.2f}+-{mf.std():4.2f}  Vacant={vi.mean():5.1f}+-{vi.std():4.1f}  [{out["sec"]}s]',flush=True)
    return out

R=[]
print('\n=== UPGRADE SUITE (3 spatial splits each) ===',flush=True)
R.append(run('E0 baseline',X32,Y32,blk32))
R.append(run('E1 +balanced sampling',X32,Y32,blk32,balanced=True))
R.append(run('E2 +pretrained ResNet18',X32,Y32,blk32,balanced=True,pre=True))
if use64:
    R.append(run('E3 +context 64px',X64,Y64,blk64,balanced=True,pre=True,crop=True))
    R.append(run('E4 +TTA',X64,Y64,blk64,balanced=True,pre=True,crop=True,tta=True))
json.dump(R,open(O/'upgrades.json','w'),indent=2)
print('\nsaved outputs/upgrades.json',flush=True)

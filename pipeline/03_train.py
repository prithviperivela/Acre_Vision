"""Ablation harness: spatially-blocked splits, class-weighted loss, honest metrics."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, sys, time
from pathlib import Path
D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs')
torch.manual_seed(0); np.random.seed(0); dev='cuda'
CH=np.load(D/'chips.npy',mmap_mode='r'); valid=np.load(D/'valid.npy')
rc=np.load(D/'rowcol.npy'); OSM=np.load(D/'osm_labels.npy',mmap_mode='r')
lp=np.load(D/'osm_label_frac.npy'); N=len(CH); eps=1e-8
BM={'blue':1,'green':2,'red':3,'nir':7,'swir1':10}
CLS=['Vacant','Urban','Water']

def spectral(idx):                     # returns rgb, indices, spectral weak label
    x=np.asarray(CH[idx],np.float32)
    r,g,b,nir,sw=[x[:,BM[k]] for k in ['red','green','blue','nir','swir1']]
    ndvi=(nir-r)/(nir+r+eps); ndwi=(nir-sw)/(nir+sw+eps); ndbi=(sw-nir)/(sw+nir+eps)
    L=np.full(ndvi.shape,3,np.int64)                       # 3 = uncertain
    L[(ndwi>0.1)&(ndvi<0.1)&(ndbi<0.0)]=2
    L[(ndbi>0.1)&(ndvi<0.25)]=1
    L[(ndvi>0.3)&(ndvi<0.6)&(ndbi<0.08)&(ndwi<0.0)]=0
    rgb=np.stack([r,g,b],1)/3000.0; ind=np.stack([ndvi,ndbi,ndwi],1)
    return np.nan_to_num(rgb),np.nan_to_num(ind),L,np.nan_to_num(x)

def blocked_split(idx,nb=8):           # contiguous spatial blocks -> no neighbour leakage
    r,c=rc[idx,0],rc[idx,1]
    br=np.digitize(r,np.quantile(r,np.linspace(0,1,nb+1)[1:-1]))
    bc=np.digitize(c,np.quantile(c,np.linspace(0,1,nb+1)[1:-1]))
    blk=br*nb+bc; ub=np.unique(blk); rng=np.random.default_rng(0); rng.shuffle(ub)
    n=len(ub); tr=set(ub[:int(.6*n)]); va=set(ub[int(.6*n):int(.8*n)])
    m=np.array([0 if x in tr else 1 if x in va else 2 for x in blk])
    return m==0,m==1,m==2

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

class PixMLP(nn.Module):
    def __init__(s,c,k=4):
        super().__init__(); s.n=nn.Sequential(nn.Conv2d(c,128,1),nn.ReLU(),nn.Conv2d(128,128,1),nn.ReLU(),nn.Conv2d(128,k,1))
    def forward(s,x): return s.n(x)

def evaluate(p,y,ncls=3,ign=255):
    """Metrics over the 3 real classes. A prediction of 'uncertain' on a real-class
    pixel counts as a FALSE NEGATIVE (an earlier version silently dropped these)."""
    lab=(y!=ign)&(y<ncls); pm,ym=p[lab],y[lab]
    f1=[];iou=[];cm=np.zeros((ncls,ncls+1),np.int64)
    for a in range(ncls):
        for bq in range(ncls+1): cm[a,bq]=int(((ym==a)&(pm==bq)).sum())
    for c in range(ncls):
        tp=int(((pm==c)&(ym==c)).sum()); fp=int(((pm==c)&(ym!=c)).sum())
        fn=int(((pm!=c)&(ym==c)).sum())
        f1.append(2*tp/max(2*tp+fp+fn,1)); iou.append(tp/max(tp+fp+fn,1))
    accm=(y!=ign)
    return dict(acc=100*float((p[accm]==y[accm]).mean()), macroF1=100*float(np.mean(f1)),
                iou={CLS[c]:round(100*iou[c],1) for c in range(ncls)}, cm=cm.tolist())

def run(name,X,Y,net_cls,ncls,ign,epochs=25,bs=64,wt=None):
    t0=time.time(); tr,va,te=blocked_split(np.arange(len(X)))
    mu=X[tr].mean((0,2,3),keepdims=True); sd=X[tr].std((0,2,3),keepdims=True)+1e-6
    Xn=((X-mu)/sd).astype(np.float32)
    xtr=torch.tensor(Xn[tr]); ytr=torch.tensor(Y[tr]); xte=torch.tensor(Xn[te]).to(dev)
    net=net_cls(X.shape[1],ncls).to(dev); opt=torch.optim.Adam(net.parameters(),2e-3)
    w=torch.tensor(wt,dtype=torch.float32,device=dev) if wt is not None else None
    for e in range(epochs):
        perm=torch.randperm(len(xtr))
        for i in range(0,len(xtr),bs):
            j=perm[i:i+bs]; xb,yb=xtr[j].to(dev),ytr[j].to(dev)
            opt.zero_grad(); F.cross_entropy(net(xb),yb,ignore_index=ign,weight=w).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p=torch.cat([net(xte[i:i+128]).argmax(1).cpu() for i in range(0,len(xte),128)]).numpy()
    m=evaluate(p.reshape(-1),Y[te].reshape(-1),3,ign); m['name']=name; m['n_train']=int(tr.sum()); m['sec']=round(time.time()-t0)
    print(f"  {name:44s} acc={m['acc']:5.2f}%  macroF1={m['macroF1']:5.2f}%  IoU={m['iou']}",flush=True)
    return m

# ---------------- data ----------------
sub=np.where(valid)[0]; rng=np.random.default_rng(0)
spec_idx=sub[rng.choice(len(sub),min(12000,len(sub)),replace=False)]
osm_idx=np.where(valid&(lp>=0.25))[0]
print(f'spectral-label subset: {len(spec_idx)} chips | OSM-label subset: {len(osm_idx)} chips\n',flush=True)
rgb_s,ind_s,Ls,x12_s=spectral(spec_idx)
rgb_o,ind_o,_,x12_o=spectral(osm_idx); Lo=np.asarray(OSM[osm_idx],np.int64)
cnt=np.array([(Lo==c).sum() for c in range(3)],float); wt=(cnt.sum()/(3*np.maximum(cnt,1)))
print('OSM class pixel counts',cnt.astype(int),'-> loss weights',wt.round(2),'\n',flush=True)

R=[]
print('== A. LEAKY: spectral weak labels, indices fed as input ==',flush=True)
R.append(run('[A1] PixelMLP on indices only (no spatial ctx)',ind_s,Ls,PixMLP,4,-1))
R.append(run('[A2] U-Net 6ch RGB+indices  <-- ORIGINAL PAPER',np.concatenate([rgb_s,ind_s],1),Ls,UNet,4,-1))
print('\n== B. HONEST-ish: spectral weak labels, indices REMOVED from input ==',flush=True)
R.append(run('[B1] U-Net 3ch RGB only',rgb_s,Ls,UNet,4,-1))
print('\n== C. NON-CIRCULAR: independent OSM labels ==',flush=True)
R.append(run('[C1] U-Net 3ch RGB only     -> OSM labels',rgb_o,Lo,UNet,3,255,wt=wt))
R.append(run('[C2] U-Net 6ch RGB+indices  -> OSM labels',np.concatenate([rgb_o,ind_o],1),Lo,UNet,3,255,wt=wt))
R.append(run('[C3] U-Net 12-band ALL      -> OSM labels  <-- MAIN',x12_o,Lo,UNet,3,255,wt=wt))
json.dump(R,open(O/'ablation.json','w'),indent=2); print('\nsaved outputs/ablation.json',flush=True)

"""Compare the v6 and v8 MODELS on an identical parcel set.

The training harness evaluates each run only on parcels its own label version
covers, so the v8 run was scored on 99 batch-2 parcels while the v6 run saw 76.
That is not a like-for-like comparison: the extra 23 are rural parcels v6's
labels never reach, and they are not the easy ones.

Both networks are ordinary CNNs and can predict on any chip regardless of which
labels trained them, so here both are run over the SAME parcels -- every
verified parcel with full 64 px context -- and scored against the same truth.
"""
import numpy as np, torch, json, sys
import segmentation_models_pytorch as smp
from pathlib import Path
D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs')
dev='cuda' if torch.cuda.is_available() else 'cpu'
if dev=='cpu': torch.set_num_threads(32)
eps=1e-8; CLS=['Vacant','Urban','Water','Other']; BM={'blue':1,'green':2,'red':3,'nir':7,'swir1':10}

def load(p):
    p=Path(p); return {int(k):v for k,v in json.load(open(p)).items() if not k.startswith('_')} if p.exists() else {}
B={'batch1':load(O/'gold'/'gold_claude.json'),'batch2':load(O/'gold2'/'gold_claude2.json'),
   'batch3':load(O/'gold3'/'gold_claude3.json')}
valid=np.load(D/'valid.npy'); off=np.load(D/'off64.npy'); rc=np.load(D/'rowcol.npy')
CH64=np.load(D/'chips64.npy',mmap_mode='r')

def feats(idx):
    x=np.asarray(CH64[idx],np.float32)
    r,g,b,nir,sw=[x[:,BM[k]] for k in ['red','green','blue','nir','swir1']]
    ndvi=(nir-r)/(nir+r+eps); ndwi=(nir-sw)/(nir+sw+eps); ndbi=(sw-nir)/(sw+nir+eps)
    return np.nan_to_num(np.concatenate([np.stack([r,g,b],1)/3000.0,np.stack([ndvi,ndbi,ndwi],1)],1))

def tta(net,x):
    acc=0
    for k in range(4):
        xr=torch.rot90(x,k,(2,3)); acc=acc+torch.rot90(net(xr).softmax(1),-k,(2,3))
        xf=torch.flip(xr,(3,)); acc=acc+torch.rot90(torch.flip(net(xf).softmax(1),(3,)),-k,(2,3))
    return acc/8

# normalisation: each model gets stats from ITS OWN label version's training pool,
# matching what it was trained under.
def norm_for(lv, tag):
    lp=np.load(D/f'osm_label_frac{lv}.npy')
    ii=np.where(valid&(lp>=0.25)&(off[:,0]==16)&(off[:,1]==16))[0]
    X=feats(ii)
    return X.mean((0,2,3),keepdims=True), X.std((0,2,3),keepdims=True)+1e-6

MODELS=[('v6','model_v6_seed{}.pt','6'),('v8','model_v8s_seed{}.pt','8')]
gold={}; [gold.update(g) for g in B.values()]
ok=np.array([g for g in sorted(gold) if valid[g] and off[g,0]==16 and off[g,1]==16])
print(f'{len(ok)} verified parcels with full 64px context (of {len(gold)})',flush=True)
truth=[gold[g] for g in ok]
Xg=feats(ok)

res={}
for name,pat,lv in MODELS:
    files=[O/pat.format(s) for s in range(3)]
    files=[f for f in files if f.exists()]
    if not files: print(f'{name}: no weights yet, skipped',flush=True); continue
    mu,sd=norm_for(lv,name)
    xg=torch.tensor(((Xg-mu)/sd).astype(np.float32))
    prob=0
    for f in files:
        net=smp.Unet('resnet18',encoder_weights=None,in_channels=6,classes=4)
        net.load_state_dict(torch.load(f,map_location='cpu')); net=net.to(dev).eval()
        with torch.no_grad():
            ps=[tta(net,xg[i:i+32].to(dev))[:,:,16:48,16:48].cpu().numpy() for i in range(0,len(xg),32)]
        prob=prob+np.concatenate(ps)
    prob/=len(files)
    pred=[CLS[int(np.bincount(p.argmax(0).reshape(-1),minlength=4).argmax())] for p in prob]
    res[name]=pred
    print(f'{name}: ensembled {len(files)} seed(s)',flush=True)

print(f'\n=== IDENTICAL parcel set, identical truth (n={len(ok)}) ===')
print(f'{"model":6s} {"accuracy":>9s} {"Vacant F1":>10s} {"Urban F1":>9s} {"Water F1":>9s} {"Other F1":>9s}')
out={}
for name,pred in res.items():
    acc=100*np.mean([a==b for a,b in zip(pred,truth)])
    f1={}
    for c in CLS:
        tp=sum(1 for a,b in zip(pred,truth) if a==c and b==c)
        fp=sum(1 for a,b in zip(pred,truth) if a==c and b!=c)
        fn=sum(1 for a,b in zip(pred,truth) if a!=c and b==c)
        f1[c]=round(100*2*tp/max(2*tp+fp+fn,1),1)
    out[name]=dict(acc=acc,f1=f1)
    print(f'{name:6s} {acc:8.1f}% {f1["Vacant"]:9.1f} {f1["Urban"]:8.1f} {f1["Water"]:8.1f} {f1["Other"]:8.1f}')

for bn,bd in B.items():
    sel=[i for i,g in enumerate(ok) if g in bd]
    if not sel: continue
    print(f'\n-- {bn} (n={len(sel)}) --')
    for name,pred in res.items():
        acc=100*np.mean([pred[i]==truth[i] for i in sel])
        tp=sum(1 for i in sel if pred[i]=='Vacant' and truth[i]=='Vacant')
        fp=sum(1 for i in sel if pred[i]=='Vacant' and truth[i]!='Vacant')
        fn=sum(1 for i in sel if pred[i]!='Vacant' and truth[i]=='Vacant')
        print(f'   {name}: acc {acc:.1f}%   Vacant F1 {100*2*tp/max(2*tp+fp+fn,1):.1f}')
json.dump(out,open(O/'fair_model_compare.json','w'),indent=2)

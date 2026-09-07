"""Train the production model (6ch RGB+indices -> independent OSM labels),
save weights, and regenerate vacancy scores for every parcel."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, pandas as pd, time
from pathlib import Path
import sys; sys.path.insert(0,'/home/prithvi/AcreVision_v2/scripts')
D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs')
torch.manual_seed(0); np.random.seed(0); dev='cuda'; eps=1e-8
CH=np.load(D/'chips.npy',mmap_mode='r'); valid=np.load(D/'valid.npy')
gid=np.load(D/'gid.npy'); rc=np.load(D/'rowcol.npy')
OSM=np.load(D/'osm_labels.npy',mmap_mode='r'); lp=np.load(D/'osm_label_frac.npy')
BM={'blue':1,'green':2,'red':3,'nir':7,'swir1':10}; CLS=['Vacant','Urban','Water']

def feats(idx):
    x=np.asarray(CH[idx],np.float32)
    r,g,b,nir,sw=[x[:,BM[k]] for k in ['red','green','blue','nir','swir1']]
    ndvi=(nir-r)/(nir+r+eps); ndwi=(nir-sw)/(nir+sw+eps); ndbi=(sw-nir)/(sw+nir+eps)
    return np.nan_to_num(np.concatenate([np.stack([r,g,b],1)/3000.0,
                                         np.stack([ndvi,ndbi,ndwi],1)],1))
class UNet(nn.Module):
    def __init__(s,c,k=3):
        super().__init__()
        B=lambda i,o:nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),
                                   nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU())
        s.e1,s.e2,s.e3=B(c,32),B(32,64),B(64,128); s.d2,s.d1=B(192,64),B(96,32); s.o=nn.Conv2d(32,k,1)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(F.max_pool2d(e1,2)); e3=s.e3(F.max_pool2d(e2,2))
        d2=s.d2(torch.cat([F.interpolate(e3,scale_factor=2),e2],1))
        return s.o(s.d1(torch.cat([F.interpolate(d2,scale_factor=2),e1],1)))

tr_idx=np.where(valid&(lp>=0.25))[0]
r,c=rc[tr_idx,0],rc[tr_idx,1]; nb=8
blk=np.digitize(r,np.quantile(r,np.linspace(0,1,nb+1)[1:-1]))*nb+np.digitize(c,np.quantile(c,np.linspace(0,1,nb+1)[1:-1]))
ub=np.unique(blk); rng=np.random.default_rng(0); rng.shuffle(ub)
hold=set(ub[:int(.2*len(ub))])                       # 20% spatially-held-out blocks
is_te=np.array([b in hold for b in blk])
X=feats(tr_idx); Y=np.asarray(OSM[tr_idx],np.int64)
mu=X[~is_te].mean((0,2,3),keepdims=True); sd=X[~is_te].std((0,2,3),keepdims=True)+1e-6
Xn=((X-mu)/sd).astype(np.float32)
cnt=np.array([(Y==c_).sum() for c_ in range(3)],float); w=torch.tensor(cnt.sum()/(3*np.maximum(cnt,1)),dtype=torch.float32,device=dev)
print(f'train {(~is_te).sum()} / held-out {is_te.sum()} chips',flush=True)

net=UNet(6).to(dev); opt=torch.optim.Adam(net.parameters(),2e-3)
xtr=torch.tensor(Xn[~is_te]); ytr=torch.tensor(Y[~is_te])
for e in range(40):
    perm=torch.randperm(len(xtr))
    for i in range(0,len(xtr),64):
        j=perm[i:i+64]; opt.zero_grad()
        F.cross_entropy(net(xtr[j].to(dev)),ytr[j].to(dev),ignore_index=255,weight=w).backward(); opt.step()
    if e%10==9: print(f'  epoch {e+1}/40',flush=True)
net.eval()
xte=torch.tensor(Xn[is_te]).to(dev)
with torch.no_grad(): p=torch.cat([net(xte[i:i+128]).argmax(1).cpu() for i in range(0,len(xte),128)]).numpy()
yt=Y[is_te].reshape(-1); pt=p.reshape(-1); lab=(yt!=255)
f1=[];iou=[]
for c_ in range(3):
    tp=int(((pt==c_)&(yt==c_)&lab).sum()); fp=int(((pt==c_)&(yt!=c_)&lab).sum()); fn=int(((pt!=c_)&(yt==c_)).sum())
    f1.append(2*tp/max(2*tp+fp+fn,1)); iou.append(tp/max(tp+fp+fn,1))
final=dict(acc=100*float((pt[lab]==yt[lab]).mean()),macroF1=100*float(np.mean(f1)),
           iou={CLS[i]:round(100*iou[i],1) for i in range(3)})
print('\nFINAL held-out:',json.dumps(final),flush=True)
torch.save({'state':net.state_dict(),'mu':mu,'sd':sd,'metrics':final},O/'model_osm_6ch.pt')

# ---- inference over every valid parcel ----
all_idx=np.where(valid)[0]; scores=np.zeros(len(all_idx)); urb=np.zeros(len(all_idx)); wat=np.zeros(len(all_idx))
for i in range(0,len(all_idx),512):
    ch=all_idx[i:i+512]; xb=torch.tensor(((feats(ch)-mu)/sd).astype(np.float32)).to(dev)
    with torch.no_grad(): pr=net(xb).argmax(1).cpu().numpy()
    scores[i:i+512]=(pr==0).mean((1,2))*100; urb[i:i+512]=(pr==1).mean((1,2))*100; wat[i:i+512]=(pr==2).mean((1,2))*100
df=pd.DataFrame({'original_chip_id':gid[all_idx],'vacancy_score':scores,
                 'urban_percentage':urb,'water_percentage':wat})
df.to_csv(O/'vacancy_scores_v2.csv',index=False)
print(f'\nwrote {len(df)} parcel scores -> outputs/vacancy_scores_v2.csv')
print(df.vacancy_score.describe().round(2).to_string())
old=pd.read_csv('/home/prithvi/osm_integration /vacancy_scores_valid_chips.csv')
m=df.merge(old[['original_chip_id','vacancy_score']],on='original_chip_id',suffixes=('_new','_old'))
print(f'\noverlap {len(m)} parcels | corr(new,old) = {m.vacancy_score_new.corr(m.vacancy_score_old):.3f}')
print(f'old mean {m.vacancy_score_old.mean():.2f}%  new mean {m.vacancy_score_new.mean():.2f}%')

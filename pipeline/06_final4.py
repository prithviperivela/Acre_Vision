"""Production model: 6ch (RGB+indices) -> independent 4-class OSM labels.

Changes vs 05: 4th 'Other' class, sqrt-inverse class weights (the raw inverse
weights drove massive Vacant over-prediction), and 3 spatial splits so the
reported numbers carry a variance, not a single lucky draw.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, pandas as pd
from pathlib import Path

D = Path('/home/prithvi/AcreVision_v2/data')
O = Path('/home/prithvi/AcreVision_v2/outputs')
dev = 'cuda'; eps = 1e-8
CH = np.load(D/'chips.npy', mmap_mode='r')
valid = np.load(D/'valid.npy'); gid = np.load(D/'gid.npy'); rc = np.load(D/'rowcol.npy')
OSM = np.load(D/'osm_labels4.npy', mmap_mode='r')
lp = np.load(D/'osm_label_frac4.npy')
pbd = np.load(D/'parcel_building_density.npy')
BM = {'blue':1,'green':2,'red':3,'nir':7,'swir1':10}
CLS = ['Vacant','Urban','Water','Other']


def feats(idx):
    x = np.asarray(CH[idx], np.float32)
    r,g,b,nir,sw = [x[:,BM[k]] for k in ['red','green','blue','nir','swir1']]
    ndvi=(nir-r)/(nir+r+eps); ndwi=(nir-sw)/(nir+sw+eps); ndbi=(sw-nir)/(sw+nir+eps)
    return np.nan_to_num(np.concatenate(
        [np.stack([r,g,b],1)/3000.0, np.stack([ndvi,ndbi,ndwi],1)], 1))


class UNet(nn.Module):
    def __init__(s, c, k=4):
        super().__init__()
        B = lambda i,o: nn.Sequential(
            nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU())
        s.e1,s.e2,s.e3 = B(c,32),B(32,64),B(64,128)
        s.d2,s.d1 = B(192,64),B(96,32); s.o = nn.Conv2d(32,k,1)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(F.max_pool2d(e1,2)); e3=s.e3(F.max_pool2d(e2,2))
        d2=s.d2(torch.cat([F.interpolate(e3,scale_factor=2),e2],1))
        return s.o(s.d1(torch.cat([F.interpolate(d2,scale_factor=2),e1],1)))


idx = np.where(valid & (lp >= 0.25))[0]
X = feats(idx); Y = np.asarray(OSM[idx], np.int64)
r, c = rc[idx,0], rc[idx,1]; nb = 8
blk = (np.digitize(r, np.quantile(r, np.linspace(0,1,nb+1)[1:-1])) * nb
       + np.digitize(c, np.quantile(c, np.linspace(0,1,nb+1)[1:-1])))
cnt = np.array([(Y==k).sum() for k in range(4)], float)
w_np = np.sqrt(cnt.sum()/(4*np.maximum(cnt,1)))          # sqrt-inverse: milder
print(f'{len(idx)} chips | class pixels {cnt.astype(int)} | weights {w_np.round(2)}', flush=True)

runs = []
for seed in range(3):
    ub = np.unique(blk); rng = np.random.default_rng(seed); rng.shuffle(ub)
    hold = set(ub[:max(1,int(.2*len(ub)))])
    te = np.array([b in hold for b in blk]); trn = ~te
    mu = X[trn].mean((0,2,3),keepdims=True); sd = X[trn].std((0,2,3),keepdims=True)+1e-6
    Xn = ((X-mu)/sd).astype(np.float32)
    net = UNet(6).to(dev); opt = torch.optim.Adam(net.parameters(), 2e-3)
    w = torch.tensor(w_np, dtype=torch.float32, device=dev)
    xtr = torch.tensor(Xn[trn]); ytr = torch.tensor(Y[trn])
    for e in range(30):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 64):
            j = perm[i:i+64]; opt.zero_grad()
            F.cross_entropy(net(xtr[j].to(dev)), ytr[j].to(dev),
                            ignore_index=255, weight=w).backward()
            opt.step()
    net.eval()
    xte = torch.tensor(Xn[te]).to(dev)
    with torch.no_grad():
        p = torch.cat([net(xte[i:i+128]).argmax(1).cpu()
                       for i in range(0,len(xte),128)]).numpy()
    yt, pt = Y[te].reshape(-1), p.reshape(-1); lab = yt != 255
    iou = []
    for k in range(4):
        tp=int(((pt==k)&(yt==k)&lab).sum()); fp=int(((pt==k)&(yt!=k)&lab).sum())
        fn=int(((pt!=k)&(yt==k)).sum()); iou.append(100*tp/max(tp+fp+fn,1))
    f1 = []
    for k in range(4):
        tp=int(((pt==k)&(yt==k)&lab).sum()); fp=int(((pt==k)&(yt!=k)&lab).sum())
        fn=int(((pt!=k)&(yt==k)).sum()); f1.append(100*2*tp/max(2*tp+fp+fn,1))
    runs.append(dict(seed=seed, acc=100*float((pt[lab]==yt[lab]).mean()),
                     macroF1=float(np.mean(f1)), iou={CLS[k]:round(iou[k],1) for k in range(4)}))
    print(f"  seed{seed}: acc={runs[-1]['acc']:.2f}%  macroF1={runs[-1]['macroF1']:.2f}%  IoU={runs[-1]['iou']}", flush=True)
    if seed == 0:
        torch.save({'state':net.state_dict(),'mu':mu,'sd':sd}, O/'model_osm4_6ch.pt')
        best = (net, mu, sd)

acc = np.array([r_['acc'] for r_ in runs]); mf = np.array([r_['macroF1'] for r_ in runs])
vi = np.array([r_['iou']['Vacant'] for r_ in runs])
print(f'\nACROSS 3 SPATIAL SPLITS:  acc {acc.mean():.2f}+-{acc.std():.2f}%   '
      f'macroF1 {mf.mean():.2f}+-{mf.std():.2f}%   Vacant IoU {vi.mean():.1f}+-{vi.std():.1f}', flush=True)

# ---- inference over every valid parcel ----
net, mu, sd = best; net.eval()
all_idx = np.where(valid)[0]
sc = np.zeros((len(all_idx), 4))
for i in range(0, len(all_idx), 512):
    ch = all_idx[i:i+512]
    xb = torch.tensor(((feats(ch)-mu)/sd).astype(np.float32)).to(dev)
    with torch.no_grad(): pr = net(xb).argmax(1).cpu().numpy()
    for k in range(4): sc[i:i+512,k] = (pr==k).mean((1,2))*100
df = pd.DataFrame({'original_chip_id':gid[all_idx], 'vacancy_score':sc[:,0],
                   'urban_percentage':sc[:,1], 'water_percentage':sc[:,2],
                   'other_percentage':sc[:,3], 'building_density':pbd[all_idx]})
df.to_csv(O/'vacancy_scores_v2.csv', index=False)
print(f'\nwrote {len(df)} parcels. mean vacancy {df.vacancy_score.mean():.2f}%', flush=True)

# ---- independent sanity check: vacancy vs OSM building density ----
old = pd.read_csv('/home/prithvi/osm_integration /vacancy_scores_valid_chips.csv')
m = df.merge(old[['original_chip_id','vacancy_score']], on='original_chip_id',
             suffixes=('_new','_old'))
print(f'\nINDEPENDENT CHECK -- corr(vacancy, building density), {len(m)} parcels')
print(f'  OLD (leaky, spectral-threshold) : {m.vacancy_score_old.corr(m.building_density):+.3f}')
print(f'  NEW (OSM-supervised)            : {m.vacancy_score_new.corr(m.building_density):+.3f}')
print('  (a valid vacancy score must be NEGATIVELY correlated with built-up density)')
print(f'  corr(new, old) = {m.vacancy_score_new.corr(m.vacancy_score_old):+.3f}')
json.dump(runs, open(O/'final4_metrics.json','w'), indent=2)

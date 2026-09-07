"""Post-hoc quality check on the v5 interstitial Vacant rule.

IMPORTANT: the spectral indices are used here ONLY as a diagnostic to audit a
label rule that was built entirely from OSM vectors.  They are not used to
create or tune any label -- doing that would reintroduce the circularity v2
removed.  Read this as "do the new labels look like the old ones?", nothing more.
"""
import numpy as np, json
from pathlib import Path
D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs')
eps=1e-8; BM={'blue':1,'green':2,'red':3,'nir':7,'swir1':10}

L4=np.load(D/'osm_labels4.npy',mmap_mode='r'); L5=np.load(D/'osm_labels5.npy',mmap_mode='r')
lp5=np.load(D/'osm_label_frac5.npy'); valid=np.load(D/'valid.npy')
CH=np.load(D/'chips.npy',mmap_mode='r')
idx=np.where(valid&(lp5>=0.25))[0]
print(f'auditing {len(idx)} train-eligible chips',flush=True)

# sample chips to keep memory sane
rng=np.random.default_rng(0); sub=rng.choice(idx,min(4000,len(idx)),replace=False); sub.sort()
x=np.asarray(CH[sub],np.float32)
r,g,b,nir,sw=[x[:,BM[k]] for k in ['red','green','blue','nir','swir1']]
ndvi=np.nan_to_num((nir-r)/(nir+r+eps)); ndbi=np.nan_to_num((sw-nir)/(sw+nir+eps))
a4=np.asarray(L4[sub]); a5=np.asarray(L5[sub])

groups={
 'Vacant (landuse tag, in v4)': (a5==0)&(a4==0),
 'Vacant (NEW interstitial)'  : (a5==0)&(a4!=0),
 'Urban'                      : (a5==1),
 'Water'                      : (a5==2),
 'Other (forest/park)'        : (a5==3),
}
print(f'\n{"group":30s} {"n_px":>9s} {"NDVI":>14s} {"NDBI":>14s}')
res={}
for k,m in groups.items():
    n=int(m.sum())
    if n==0: continue
    v,d=ndvi[m],ndbi[m]
    res[k]=dict(n=n,ndvi=[float(v.mean()),float(v.std())],ndbi=[float(d.mean()),float(d.std())])
    print(f'{k:30s} {n:9d} {v.mean():7.3f}+-{v.std():5.3f} {d.mean():7.3f}+-{d.std():5.3f}')

# how separable is NEW-vacant from Urban vs from old-Vacant? (Cohen's d on NDBI)
def cohen(a,b):
    return abs(a[0]-b[0])/np.sqrt((a[1]**2+b[1]**2)/2+1e-9)
nv,ov,ub=res['Vacant (NEW interstitial)'],res['Vacant (landuse tag, in v4)'],res['Urban']
print(f'\nNDBI separation of NEW-vacant from:')
print(f'  old Vacant  d={cohen(nv["ndbi"],ov["ndbi"]):.2f}   (want SMALL -> same concept)')
print(f'  Urban       d={cohen(nv["ndbi"],ub["ndbi"]):.2f}   (want LARGE -> not mislabelled built-up)')

# per-chip composition of the newly added pixels
newpx=(a5==0)&(a4!=0)
frac=newpx.reshape(len(sub),-1).mean(1)
print(f'\nchips with >50% newly-Vacant pixels: {(frac>0.5).sum()}/{len(sub)}')
json.dump(res,open(O/'label_audit.json','w'),indent=2)

# ---- visual contact sheet: 24 chips most dominated by NEW vacant ----
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
top=sub[np.argsort(-frac)[:24]]
xt=np.asarray(CH[top],np.float32); a5t=np.asarray(L5[top]); a4t=np.asarray(L4[top])
fig,axes=plt.subplots(4,12,figsize=(26,9))
for j,i in enumerate(range(24)):
    rgb=np.stack([xt[i,3],xt[i,2],xt[i,1]],-1)
    rgb=np.nan_to_num(np.clip(rgb/np.nanpercentile(rgb,98),0,1))
    cr,cc=divmod(j,6); ax=axes[cr,cc*2]
    ax.imshow(rgb,interpolation='nearest'); ax.axis('off')
    ax.set_title(f'gid {top[i]}',fontsize=7)
    ov=np.zeros((32,32,3)); nm=(a5t[i]==0)&(a4t[i]!=0); om=(a5t[i]==0)&(a4t[i]==0)
    ov[...,0]=nm; ov[...,1]=om; ov[...,2]=(a5t[i]==1)
    ax2=axes[cr,cc*2+1]; ax2.imshow(rgb); ax2.imshow(ov,alpha=0.45); ax2.axis('off')
    ax2.set_title('R=new G=old B=urban',fontsize=6)
plt.tight_layout(); plt.savefig(O/'label_audit.png',dpi=95); print(f'\nwrote {O}/label_audit.png')

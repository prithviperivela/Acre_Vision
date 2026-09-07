"""Second, non-overlapping verified batch.

Two jobs:
 1. Unlock the v7 minor-road gate. Batch 1 is where the gate's threshold gets
    fitted, so batch 1 can no longer measure it. Batch 2 is untouched by that
    fitting and becomes the honest test.
 2. Close the biggest evaluation gap in the last report: batch 1 was drawn only
    from OSM-labelled parcels, so it said nothing about the 17,921 UNLABELLED
    ones that dominate the city-wide mean. Batch 2 adds them as a stratum.
"""
import numpy as np, json, math, time, urllib.request, concurrent.futures as cf
from pathlib import Path
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs')
G=O/'gold'; G2=O/'gold2'; G2.mkdir(exist_ok=True); (G2/'sheets').mkdir(exist_ok=True)
TC=G/'_tiles'   # share the tile cache with batch 1
TIF='/home/prithvi/Downloads/Hyderabad_Sentinel2_12Bands_JanMar2025.tif'
URL=('https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery'
     '/MapServer/tile/{z}/{y}/{x}')
Z=18; CS=32

with rasterio.open(TIF) as s: tr,crs=s.transform,s.crs
to_wgs=Transformer.from_crs(crs,'EPSG:4326',always_xy=True)
L4=np.load(D/'osm_labels4.npy',mmap_mode='r'); L5=np.load(D/'osm_labels5.npy',mmap_mode='r')
L6=np.load(D/'osm_labels6.npy',mmap_mode='r')
lp=np.load(D/'osm_label_frac6.npy'); valid=np.load(D/'valid.npy')
rc=np.load(D/'rowcol.npy'); off=np.load(D/'off64.npy')

batch1={int(k) for k in json.load(open(G/'gold_claude.json')) if not k.startswith('_')}
lab=np.where(valid&(lp>=0.25)&(off[:,0]==16)&(off[:,1]==16))[0]
unl=np.where(valid&(lp<0.25)&(off[:,0]==16)&(off[:,1]==16))[0]
lab=np.array([i for i in lab if i not in batch1])
unl=np.array([i for i in unl if i not in batch1])

a5=np.asarray(L5[lab]); a4=np.asarray(L4[lab]); a6=np.asarray(L6[lab])
n=len(lab)
f_new=((a5==0)&(a4!=0)).reshape(n,-1).mean(1)
f_old=((a5==0)&(a4==0)).reshape(n,-1).mean(1)
f={k:(a6.reshape(n,-1)==k).mean(1) for k in range(4)}
rng=np.random.default_rng(2024)
strata={'interstitial':(lab[f_new>0.5],40),'vacant_tag':(lab[f_old>0.5],12),
        'urban':(lab[f[1]>0.7],12),'water':(lab[f[2]>0.7],6),'other':(lab[f[3]>0.7],6),
        'unlabelled':(unl,24)}
sample=[]
for name,(pool,k) in strata.items():
    take=rng.choice(pool,min(k,len(pool)),replace=False)
    print(f'  {name:13s} pool={len(pool):6d} take={len(take)}',flush=True)
    for g in take:
        g=int(g); r,c=int(rc[g,0]),int(rc[g,1])
        X,Y=xy(tr,r+CS/2,c+CS/2); lon,lat=to_wgs.transform(X,Y)
        sample.append(dict(gid=g,stratum=name,lat=float(lat),lon=float(lon),row=r,col=c))
print(f'total {len(sample)} chips (batch 1 had {len(batch1)}), zero overlap',flush=True)

def deg2num(lat,lon,z):
    nn=2.0**z
    return ((lon+180.0)/360.0*nn,(1.0-math.asinh(math.tan(math.radians(lat)))/math.pi)/2.0*nn)
def fetch(z,x,y):
    p=TC/f'{z}_{y}_{x}.jpg'
    if p.exists(): return p
    req=urllib.request.Request(URL.format(z=z,x=x,y=y),
                               headers={'User-Agent':'AcreVision-research/1.0'})
    for a in range(3):
        try:
            with urllib.request.urlopen(req,timeout=25) as r: p.write_bytes(r.read())
            return p
        except Exception: time.sleep(1.5*(a+1))
    return None
def highres(lat,lon,span=320.0):
    dlat=span/2/111320.0; dlon=span/2/(111320.0*math.cos(math.radians(lat)))
    x0f,y0f=deg2num(lat+dlat,lon-dlon,Z); x1f,y1f=deg2num(lat-dlat,lon+dlon,Z)
    X0,Y0,X1,Y1=int(x0f),int(y0f),int(x1f),int(y1f)
    mos=Image.new('RGB',((X1-X0+1)*256,(Y1-Y0+1)*256))
    jobs=[(X0+i,Y0+j) for i in range(X1-X0+1) for j in range(Y1-Y0+1)]
    with cf.ThreadPoolExecutor(8) as ex:
        for (x,y),p in zip(jobs,ex.map(lambda t:fetch(Z,t[0],t[1]),jobs)):
            if p:
                try: mos.paste(Image.open(p).convert('RGB'),((x-X0)*256,(y-Y0)*256))
                except Exception: pass
    l,t=int((x0f-X0)*256),int((y0f-Y0)*256); r,b=int((x1f-X0)*256),int((y1f-Y0)*256)
    return np.asarray(mos.crop((l,t,max(r,l+16),max(b,t+16))))

# pre-fetch every tile, then render BLIND sheets (imagery only, shuffled)
for k,s in enumerate(sample):
    highres(s['lat'],s['lon'])
    if (k+1)%20==0: print(f'  fetched {k+1}/{len(sample)}',flush=True)
json.dump(sample,open(G2/'sample.json','w'),indent=1)

rng2=np.random.default_rng(555)
shuf=[sample[i] for i in rng2.permutation(len(sample))]
json.dump([s['gid'] for s in shuf],open(G2/'blind_order.json','w'))
PER=6
for k in range(0,len(shuf),PER):
    grp=shuf[k:k+PER]
    fig,ax=plt.subplots(2,3,figsize=(16.5,11.6))
    for j,s in enumerate(grp):
        a=ax[j//3,j%3]
        try: im=highres(s['lat'],s['lon'])
        except Exception: im=np.zeros((16,16,3),np.uint8)
        a.imshow(im); a.axis('off'); a.set_title(f'gid {s["gid"]}',fontsize=15,fontweight='bold')
    for j in range(len(grp),6): ax[j//3,j%3].axis('off')
    fig.suptitle(f'BATCH 2  sheet {k//PER:02d}   320 m across each   Esri z18 ~0.57 m/px\n'
                 'classify centre-dominant land: Vacant(developable open) / Urban / Water / Other(forest,park,infrastructure)',
                 fontsize=12)
    plt.tight_layout(); plt.savefig(G2/'sheets'/f'sheet_{k//PER:02d}.png',dpi=78); plt.close(fig)
print(f'wrote {math.ceil(len(shuf)/PER)} sheets -> {G2}/sheets',flush=True)
print('DONE',flush=True)

"""Gold batch 3 -- the precision test of the v8 rural-open rule.

v8 claims 65.9% of the AOI as Vacant, up from v6's 4.0%.  That is either the
correct answer (most of a 3,090 km2 AOI really is open rural land) or a repeat
of the v3 failure, where every unmapped rural pixel became false Vacant.

Batch 3 samples ONLY from the land v8 newly claims and v6 did not, so its
Vacant precision is measured directly.  No overlap with batches 1 or 2.
"""
import numpy as np, json, math, time, urllib.request, concurrent.futures as cf
from pathlib import Path
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs')
G3=O/'gold3'; G3.mkdir(exist_ok=True); (G3/'sheets').mkdir(exist_ok=True)
TC=O/'gold'/'_tiles'
TIF='/home/prithvi/Downloads/Hyderabad_Sentinel2_12Bands_JanMar2025.tif'
URL=('https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery'
     '/MapServer/tile/{z}/{y}/{x}')
Z=18; CS=32; N=60
with rasterio.open(TIF) as s: tr,crs=s.transform,s.crs
to_wgs=Transformer.from_crs(crs,'EPSG:4326',always_xy=True)
rc=np.load(D/'rowcol.npy'); valid=np.load(D/'valid.npy'); off=np.load(D/'off64.npy')
newopen=np.load(D/'v8_new_open.npy')          # pixels v8 claims that v6 did not

prev=set()
for p in [O/'gold'/'gold_claude.json', O/'gold2'/'gold_claude2.json']:
    if p.exists(): prev|={int(k) for k in json.load(open(p)) if not k.startswith('_')}

cand=np.where(valid&(off[:,0]==16)&(off[:,1]==16))[0]
frac=np.array([newopen[rc[g,0]:rc[g,0]+CS, rc[g,1]:rc[g,1]+CS].mean() for g in cand])
pool=cand[(frac>0.6)]
pool=np.array([g for g in pool if g not in prev])
print(f'{len(pool)} parcels are >60% newly-claimed-by-v8 (excluding batches 1-2)',flush=True)
rng=np.random.default_rng(31337)
take=rng.choice(pool,min(N,len(pool)),replace=False)
sample=[]
for g in take:
    g=int(g); r,c=int(rc[g,0]),int(rc[g,1])
    X,Y=xy(tr,r+CS/2,c+CS/2); lon,lat=to_wgs.transform(X,Y)
    sample.append(dict(gid=g,stratum='v8_new_open',lat=float(lat),lon=float(lon),row=r,col=c))
print(f'sampled {len(sample)}',flush=True)

def deg2num(lat,lon,z):
    n=2.0**z
    return ((lon+180.0)/360.0*n,(1.0-math.asinh(math.tan(math.radians(lat)))/math.pi)/2.0*n)
def fetch(z,x,y):
    p=TC/f'{z}_{y}_{x}.jpg'
    if p.exists(): return p
    req=urllib.request.Request(URL.format(z=z,x=x,y=y),headers={'User-Agent':'AcreVision-research/1.0'})
    for a in range(3):
        try:
            with urllib.request.urlopen(req,timeout=25) as r: p.write_bytes(r.read()); return p
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

for k,s in enumerate(sample):
    highres(s['lat'],s['lon'])
    if (k+1)%20==0: print(f'  fetched {k+1}/{len(sample)}',flush=True)
json.dump(sample,open(G3/'sample.json','w'),indent=1)
rng2=np.random.default_rng(777)
shuf=[sample[i] for i in rng2.permutation(len(sample))]
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
    fig.suptitle(f'BATCH 3  sheet {k//PER:02d}   320 m across each   Esri z18 ~0.57 m/px\n'
                 'classify centre-dominant land: Vacant(developable open) / Urban / Water / Other(forest,park,infrastructure)',
                 fontsize=12)
    plt.tight_layout(); plt.savefig(G3/'sheets'/f'sheet_{k//PER:02d}.png',dpi=78); plt.close(fig)
print(f'wrote {math.ceil(len(shuf)/PER)} sheets',flush=True); print('DONE',flush=True)

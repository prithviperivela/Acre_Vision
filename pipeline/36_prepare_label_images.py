"""Render all 256 verified parcels as high-resolution PNGs for human labelling.

The tiles are already cached from the fetch scripts, so this is offline. Each
image is the same 320 m footprint the model is scored on, at Esri z18 (~0.57
m/px) -- the resolution at which "roof or bare ground?" is actually decidable.
The 10 m Sentinel view is deliberately NOT shown: it is too coarse to judge, and
showing it would bias the annotator toward what the model can see.
"""
import numpy as np, json, math
from pathlib import Path
from PIL import Image
O=Path('/home/prithvi/AcreVision_v2/outputs'); TC=O/'gold'/'_tiles'
OUT=O/'label_images'; OUT.mkdir(exist_ok=True)
Z=18

def deg2num(lat,lon,z):
    n=2.0**z
    return ((lon+180.0)/360.0*n,(1.0-math.asinh(math.tan(math.radians(lat)))/math.pi)/2.0*n)
def highres(lat,lon,span=320.0):
    dlat=span/2/111320.0; dlon=span/2/(111320.0*math.cos(math.radians(lat)))
    x0f,y0f=deg2num(lat+dlat,lon-dlon,Z); x1f,y1f=deg2num(lat-dlat,lon+dlon,Z)
    X0,Y0,X1,Y1=int(x0f),int(y0f),int(x1f),int(y1f)
    mos=Image.new('RGB',((X1-X0+1)*256,(Y1-Y0+1)*256))
    miss=0
    for x in range(X0,X1+1):
        for y in range(Y0,Y1+1):
            p=TC/f'{Z}_{y}_{x}.jpg'
            if p.exists():
                try: mos.paste(Image.open(p).convert('RGB'),((x-X0)*256,(y-Y0)*256))
                except Exception: miss+=1
            else: miss+=1
    l,t=int((x0f-X0)*256),int((y0f-Y0)*256); r,b=int((x1f-X0)*256),int((y1f-Y0)*256)
    return mos.crop((l,t,max(r,l+16),max(b,t+16))), miss

samples=[]
for sub,f in [('gold','sample.json'),('gold2','sample.json'),('gold3','sample.json')]:
    p=O/sub/f
    if p.exists(): samples += json.load(open(p))
seen=set(); uniq=[]
for s in samples:
    if s['gid'] not in seen: seen.add(s['gid']); uniq.append(s)
print(f'{len(uniq)} unique verified parcels',flush=True)

rng=np.random.default_rng(4242)                 # one fixed presentation order for everyone
order=[uniq[i] for i in rng.permutation(len(uniq))]
json.dump([{'gid':s['gid'],'lat':s['lat'],'lon':s['lon'],'stratum':s['stratum']} for s in order],
          open(OUT/'order.json','w'),indent=1)

bad=0
for k,s in enumerate(order):
    fp=OUT/f'{s["gid"]}.jpg'
    if fp.exists(): continue
    im,miss=highres(s['lat'],s['lon'])
    if miss: bad+=1
    im.resize((640,640),Image.LANCZOS).save(fp,quality=88)
    if (k+1)%50==0: print(f'  {k+1}/{len(order)}',flush=True)
print(f'wrote {len(order)} images -> {OUT}  ({bad} had missing tiles)',flush=True)
print('DONE',flush=True)

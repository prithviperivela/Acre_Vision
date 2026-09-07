"""Figure for the report: what changed between v4 and v6 labels, on gold chips."""
import numpy as np, json, math
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs'); G=O/'gold'
TC=G/'_tiles'; Z=18
CLS=['Vacant','Urban','Water','Other']
cmap=ListedColormap(['#d97706','#6b7280','#2563eb','#15803d'])   # vacant/urban/water/other

gold={int(k):v for k,v in json.load(open(G/'gold_claude.json')).items() if not k.startswith('_')}
sample={s['gid']:s for s in json.load(open(G/'sample.json'))}
L4=np.load(D/'osm_labels4.npy',mmap_mode='r'); L6=np.load(D/'osm_labels6.npy',mmap_mode='r')

def deg2num(lat,lon,z):
    n=2.0**z
    return ((lon+180.0)/360.0*n,(1.0-math.asinh(math.tan(math.radians(lat)))/math.pi)/2.0*n)
def hr(lat,lon,span=320.0):
    dlat=span/2/111320.0; dlon=span/2/(111320.0*math.cos(math.radians(lat)))
    x0f,y0f=deg2num(lat+dlat,lon-dlon,Z); x1f,y1f=deg2num(lat-dlat,lon+dlon,Z)
    X0,Y0,X1,Y1=int(x0f),int(y0f),int(x1f),int(y1f)
    m=Image.new('RGB',((X1-X0+1)*256,(Y1-Y0+1)*256))
    for x in range(X0,X1+1):
        for y in range(Y0,Y1+1):
            p=TC/f'{Z}_{y}_{x}.jpg'
            if p.exists(): m.paste(Image.open(p).convert('RGB'),((x-X0)*256,(y-Y0)*256))
    l,t=int((x0f-X0)*256),int((y0f-Y0)*256); r,b=int((x1f-X0)*256),int((y1f-Y0)*256)
    return np.asarray(m.crop((l,t,max(r,l+16),max(b,t+16))))

def maj(a):
    a=np.asarray(a); a=a[a!=255]
    return CLS[int(np.bincount(a,minlength=4).argmax())] if a.size else '-'

# pick chips where v6 fixed v4: gold=Vacant that v4 missed, and gold=Other that v4 called Vacant
fixed_v=[g for g,v in gold.items() if v=='Vacant' and maj(L4[g])!='Vacant' and maj(L6[g])=='Vacant']
fixed_o=[g for g,v in gold.items() if v=='Other'  and maj(L4[g])=='Vacant' and maj(L6[g])=='Other']
pick=fixed_v[:4]+fixed_o[:4]
print(f'{len(fixed_v)} chips v6 recovered as Vacant, {len(fixed_o)} scrub chips v6 moved to Other')

fig,ax=plt.subplots(3,len(pick),figsize=(3.0*len(pick),9.4))
for j,g in enumerate(pick):
    s=sample[g]
    ax[0,j].imshow(hr(s['lat'],s['lon'])); ax[0,j].set_title(f'gold = {gold[g]}',fontsize=11,fontweight='bold')
    ax[1,j].imshow(np.where(np.asarray(L4[g])==255,np.nan,np.asarray(L4[g])),cmap=cmap,vmin=0,vmax=3,interpolation='nearest')
    ax[1,j].set_title(f'v4 -> {maj(L4[g])}',fontsize=10)
    ax[2,j].imshow(np.where(np.asarray(L6[g])==255,np.nan,np.asarray(L6[g])),cmap=cmap,vmin=0,vmax=3,interpolation='nearest')
    ax[2,j].set_title(f'v6 -> {maj(L6[g])}',fontsize=10)
    for r in range(3): ax[r,j].set_xticks([]); ax[r,j].set_yticks([])
for r,lab in enumerate(['Esri z18 imagery','old labels (v4)','new labels (v6)']):
    ax[r,0].set_ylabel(lab,fontsize=11,fontweight='bold')
fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in
                    zip(['#d97706','#6b7280','#2563eb','#15803d'],CLS)]+
                   [Patch(facecolor='white',edgecolor='#bbb',label='unlabelled (ignore)')],
           loc='lower center',ncol=5,frameon=False,fontsize=10)
fig.suptitle('What the label fix changed  |  left 4: vacant land v4 missed   right 4: OSM "scrub" that is really woodland',
             fontsize=12)
plt.tight_layout(rect=[0,0.045,1,0.97]); plt.savefig(O/'label_fix_figure.png',dpi=105)
print('wrote outputs/label_fix_figure.png')

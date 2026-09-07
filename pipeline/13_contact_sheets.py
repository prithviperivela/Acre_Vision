"""Blind labelling sheets: high-resolution imagery ONLY.

The OSM v5 mask is deliberately withheld here.  The point of the gold set is to
audit that mask, so the annotator must not see it while deciding -- otherwise the
'gold' labels inherit the very bias being measured.  Chips are shuffled so the
stratum cannot be inferred from position either.

6 chips per sheet -> outputs/gold/sheets/sheet_NN.png
"""
import numpy as np, json, math
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
import rasterio
from rasterio.transform import xy
from pyproj import Transformer

G = Path('/home/prithvi/AcreVision_v2/outputs/gold'); S = G/'sheets'; S.mkdir(exist_ok=True)
TC = G/'_tiles'
D = Path('/home/prithvi/AcreVision_v2/data')
Z = 18
URLZ = 256

sample = json.load(open(G/'sample.json'))
rng = np.random.default_rng(99)
order = rng.permutation(len(sample))
shuffled = [sample[i] for i in order]
json.dump([s['gid'] for s in shuffled], open(G/'blind_order.json', 'w'))

def deg2num(lat, lon, z):
    n = 2.0**z
    return ((lon+180.0)/360.0*n,
            (1.0-math.asinh(math.tan(math.radians(lat)))/math.pi)/2.0*n)

def highres(lat, lon, span_m=320.0):
    dlat = span_m/2/111320.0; dlon = span_m/2/(111320.0*math.cos(math.radians(lat)))
    x0f, y0f = deg2num(lat+dlat, lon-dlon, Z); x1f, y1f = deg2num(lat-dlat, lon+dlon, Z)
    X0, Y0, X1, Y1 = int(x0f), int(y0f), int(x1f), int(y1f)
    mos = Image.new('RGB', ((X1-X0+1)*URLZ, (Y1-Y0+1)*URLZ))
    for x in range(X0, X1+1):
        for y in range(Y0, Y1+1):
            p = TC/f'{Z}_{y}_{x}.jpg'
            if p.exists(): mos.paste(Image.open(p).convert('RGB'), ((x-X0)*URLZ, (y-Y0)*URLZ))
    l, t = int((x0f-X0)*URLZ), int((y0f-Y0)*URLZ)
    r, b = int((x1f-X0)*URLZ), int((y1f-Y0)*URLZ)
    return np.asarray(mos.crop((l, t, max(r, l+16), max(b, t+16))))

PER = 6
for k in range(0, len(shuffled), PER):
    grp = shuffled[k:k+PER]
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 11.6))
    for j, s in enumerate(grp):
        a = ax[j//3, j % 3]
        try: hr = highres(s['lat'], s['lon'])
        except Exception: hr = np.zeros((16, 16, 3), np.uint8)
        a.imshow(hr); a.axis('off')
        a.set_title(f'gid {s["gid"]}', fontsize=15, fontweight='bold')
    for j in range(len(grp), 6): ax[j//3, j % 3].axis('off')
    fig.suptitle(f'sheet {k//PER:02d}   320 m across each   Esri z18 ~0.57 m/px\n'
                 'classify centre-dominant land: Vacant(developable open) / Urban / Water / Other(forest,park,quarry)',
                 fontsize=12)
    plt.tight_layout(); plt.savefig(S/f'sheet_{k//PER:02d}.png', dpi=78); plt.close(fig)
print(f'wrote {math.ceil(len(shuffled)/PER)} sheets -> {S}', flush=True)

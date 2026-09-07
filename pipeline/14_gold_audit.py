"""Score the OSM label rules against the blind high-res gold set.

Decides one question: does the v5 'mapped-but-empty' interstitial rule produce
Vacant labels that are actually vacant?  Precision here is the number that
matters -- a rule that mints Vacant labels on unmapped settlements poisons the
class it was meant to rescue.
"""
import numpy as np, json
from pathlib import Path
D = Path('/home/prithvi/AcreVision_v2/data'); O = Path('/home/prithvi/AcreVision_v2/outputs')
G = O/'gold'
CLS = ['Vacant', 'Urban', 'Water', 'Other']

gold = {int(k): v for k, v in json.load(open(G/'gold_claude.json')).items() if not k.startswith('_')}
sample = {s['gid']: s for s in json.load(open(G/'sample.json'))}
L4 = np.load(D/'osm_labels4.npy', mmap_mode='r'); L5 = np.load(D/'osm_labels5.npy', mmap_mode='r')

def osm_major(arr):
    """chip-level majority OSM class, ignoring 255"""
    a = np.asarray(arr); a = a[a != 255]
    return CLS[int(np.bincount(a, minlength=4).argmax())] if a.size else None

rows = []
for g, gl in gold.items():
    rows.append(dict(gid=g, gold=gl, stratum=sample[g]['stratum'],
                     v4=osm_major(L4[g]), v5=osm_major(L5[g]),
                     f_new=sample[g]['f_new']))
scored = [r for r in rows if r['gold'] != 'Mixed']
print(f'{len(rows)} chips, {len(scored)} scorable ({len(rows)-len(scored)} Mixed excluded)\n')

print('=== PER-STRATUM AGREEMENT WITH GOLD ===')
print(f'{"stratum":14s} {"n":>3s}  {"v4 agrees":>10s}  {"v5 agrees":>10s}   gold breakdown')
for st in ['interstitial', 'vacant_tag', 'urban', 'water', 'other']:
    sub = [r for r in scored if r['stratum'] == st]
    if not sub: continue
    a4 = np.mean([r['v4'] == r['gold'] for r in sub])*100
    a5 = np.mean([r['v5'] == r['gold'] for r in sub])*100
    from collections import Counter
    bd = ', '.join(f'{k}={v}' for k, v in Counter(r['gold'] for r in sub).most_common())
    print(f'{st:14s} {len(sub):3d}  {a4:9.1f}%  {a5:9.1f}%   {bd}')

print('\n=== THE DECISIVE NUMBER: precision of each Vacant rule ===')
for tag, key, sel in [('v4  landuse-tag Vacant', 'v4', lambda r: r['v4'] == 'Vacant'),
                      ('v5  Vacant (all)',       'v5', lambda r: r['v5'] == 'Vacant'),
                      ('v5  Vacant from NEW interstitial only', 'v5',
                       lambda r: r['v5'] == 'Vacant' and r['stratum'] == 'interstitial')]:
    sub = [r for r in scored if sel(r)]
    if not sub: continue
    prec = np.mean([r['gold'] == 'Vacant' for r in sub])*100
    from collections import Counter
    err = Counter(r['gold'] for r in sub if r['gold'] != 'Vacant')
    print(f'  {tag:40s} n={len(sub):3d}  precision={prec:5.1f}%   '
          f'errors: {dict(err) if err else "none"}')

print('\n=== confusion: gold (rows) vs v5 OSM label (cols) ===')
print(f'{"":10s}' + ''.join(f'{c:>9s}' for c in CLS))
for gc in CLS:
    r = [sum(1 for x in scored if x['gold'] == gc and x['v5'] == pc) for pc in CLS]
    print(f'{gc:10s}' + ''.join(f'{v:9d}' for v in r))

json.dump(rows, open(G/'audit_rows.json', 'w'), indent=1)
print(f'\nwrote {G}/audit_rows.json')

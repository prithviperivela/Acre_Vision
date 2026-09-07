"""Attach the validated v8 vacancy model to the existing 6-slider suitability layer.

The dashboard's composite score is
    composite = SUM_i  weight_i * subscore_i / 100
over six subscores: vacancy, road, healthcare, education, lifestyle, essential.
Five of those come from OSM POI/road proximity and are untouched here. The sixth,
vacancy, was produced by the original spectral-threshold model -- the one whose
98.5% accuracy turned out to be circular, and which correlates only -0.085 with
built-up density.

This swaps in the v8 model's vacancy score, normalised the same way the original
was (min-max to 0-100, verified against the shipped CSV), and keeps the original
value alongside it so the two can be compared parcel by parcel in the dashboard.
Being able to flip between them is the point: it is how the new model gets
checked against real imagery rather than taken on trust.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

SIX = Path('/home/prithvi/osm_integration /acre_model_outputs/6slider_parcel_scores.csv')
V8 = Path('/home/prithvi/AcreVision_v2/outputs/vacancy_scores_v8.csv')
OUT = Path('/home/prithvi/AcreVision_v2/outputs/dashboard')
OUT.mkdir(exist_ok=True)

six = pd.read_csv(SIX)
v8 = pd.read_csv(V8)
print(f'6-slider layer : {len(six)} parcels')
print(f'v8 model       : {len(v8)} parcels')

d = six.merge(v8, left_on='gid', right_on='original_chip_id',
              suffixes=('_v1', '_v8'), how='inner')
print(f'merged on gid  : {len(d)}')


def minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) * 100 if hi > lo else s*0


# verify we reproduce the original normalisation before trusting it on new data
check = minmax(d['vacancy_score_v1'])
assert np.allclose(check, d['vacancy_subscore'], atol=0.1), 'normalisation mismatch'
print('normalisation reproduces the shipped vacancy_subscore  OK')

d['vac_sub_v8'] = minmax(d['vacancy_score_v8'])
d['vac_sub_v1'] = d['vacancy_subscore']

print(f"\ncorr(v1 vacancy, v8 vacancy) = {d.vacancy_score_v1.corr(d.vacancy_score_v8):+.3f}")
print(f"corr(v1, building density)  = {d.vacancy_score_v1.corr(d.building_density):+.3f}")
print(f"corr(v8, building density)  = {d.vacancy_score_v8.corr(d.building_density):+.3f}")

cols = dict(
    gid=d.gid.astype(int).tolist(),
    lat=[round(x, 5) for x in d.latitude],
    lon=[round(x, 5) for x in d.longitude],
    vac8=[round(x, 1) for x in d.vac_sub_v8],
    vac1=[round(x, 1) for x in d.vac_sub_v1],
    road=[round(x, 1) for x in d.road_subscore],
    heal=[round(x, 1) for x in d.healthcare_subscore],
    educ=[round(x, 1) for x in d.education_subscore],
    life=[round(x, 1) for x in d.lifestyle_subscore],
    esse=[round(x, 1) for x in d.essential_services_subscore],
    urb=[round(x, 1) for x in d.urban_percentage],
    wat=[round(x, 1) for x in d.water_percentage],
    oth=[round(x, 1) for x in d.other_percentage],
    bd=[round(x, 3) for x in d.building_density],
)

# mark parcels that carry a human/imagery-verified label, so the dashboard can
# show which recommendations sit on checked ground
O = Path('/home/prithvi/AcreVision_v2/outputs')
gid_arr = np.load('/home/prithvi/AcreVision_v2/data/gid.npy')
verified = {}
for p in [O/'gold'/'gold_claude.json', O/'gold2'/'gold_claude2.json', O/'gold3'/'gold_claude3.json']:
    if p.exists():
        for k, v in json.load(open(p)).items():
            if not k.startswith('_'):
                verified[int(gid_arr[int(k)])] = v
cols['ver'] = [verified.get(g, '') for g in cols['gid']]
print(f'{sum(1 for x in cols["ver"] if x)} of the dashboard parcels carry a verified label')

meta = dict(
    n=len(d),
    lat0=min(cols['lat']), lat1=max(cols['lat']),
    lon0=min(cols['lon']), lon1=max(cols['lon']),
    corr_v1=float(d.vacancy_score_v1.corr(d.building_density)),
    corr_v8=float(d.vacancy_score_v8.corr(d.building_density)),
    corr_v1_v8=float(d.vacancy_score_v1.corr(d.vacancy_score_v8)),
)
payload = dict(meta=meta, **cols)
p = OUT/'data.json'
p.write_text(json.dumps(payload, separators=(',', ':')))
print(f"\nwrote {p}  ({p.stat().st_size/1024/1024:.2f} MB)")

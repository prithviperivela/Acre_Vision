"""Assemble the public Acre_Vision repository from this working tree.

Deliberately excluded, with reasons:
  * data/*.npy          7.3 GB of chips and rasters -- regenerable by 01/08
  * outputs/*.pt        54.8 MB per seed; six of them. Retrain instructions in README
  * outputs/gold*/_tiles, outputs/label_images
                        Esri World Imagery. Fine to fetch for research, not ours
                        to redistribute. scripts/12,25,30,36 re-fetch them.
The deployable Dash app is kept working and upgraded to the v8 vacancy score
rather than removed, so the existing Render deployment keeps functioning.
"""
import shutil
import pandas as pd
from pathlib import Path

SRC = Path('/home/prithvi/AcreVision_v2')
SIX = Path('/home/prithvi/osm_integration /acre_model_outputs/6slider_parcel_scores.csv')
REPO = Path('/tmp/AV_repo')
assert REPO.exists(), 'clone the repo to /tmp/AV_repo first'

# ---- wipe tracked content, keep .git ----
for p in REPO.iterdir():
    if p.name == '.git':
        continue
    shutil.rmtree(p) if p.is_dir() else p.unlink()

for d in ['pipeline', 'labels', 'metrics', 'docs', 'data']:
    (REPO/d).mkdir(parents=True, exist_ok=True)

# ---- pipeline scripts ----
n = 0
for f in sorted((SRC/'scripts').glob('*.py')):
    shutil.copy(f, REPO/'pipeline'/f.name); n += 1
print(f'pipeline: {n} scripts')

# ---- verified label sets + the tools that make and score them ----
for sub, name in [('gold', 'gold_claude_batch1.json'), ('gold2', 'gold_claude_batch2.json'),
                  ('gold3', 'gold_claude_batch3.json')]:
    src = list((SRC/'outputs'/sub).glob('gold_claude*.json'))
    if src:
        shutil.copy(src[0], REPO/'labels'/name)
for sub, name in [('gold', 'sample_batch1.json'), ('gold2', 'sample_batch2.json'),
                  ('gold3', 'sample_batch3.json')]:
    p = SRC/'outputs'/sub/'sample.json'
    if p.exists():
        shutil.copy(p, REPO/'labels'/name)
p = SRC/'outputs'/'label_images'/'order.json'
if p.exists():
    shutil.copy(p, REPO/'labels'/'label_order.json')
print(f'labels: {len(list((REPO/"labels").iterdir()))} files')

# ---- metrics ----
keep = ['final_v8s.json', 'final_v6.json', 'final_v4.json', 'label_scoreboard.json',
        'developability_audit.json', 'fair_model_compare.json', 'v7_validation.json',
        'v7_gate.json', 'labels5.json', 'labels6.json', 'labels7.json', 'labels8.json',
        'scores_v8_summary.json', 'gold_strata.json', 'label_audit.json',
        'v2_vs_gold.json', 'v7_proposal.json', 'upgrades.json', 'ablation.json']
m = 0
for k in keep:
    p = SRC/'outputs'/k
    if p.exists():
        shutil.copy(p, REPO/'metrics'/k); m += 1
for k in ['32_scoreboard.log', '28_validate.log', '15_fclass_blame.log',
          '33_city_weighted.log', '39_city_model_acc.log', '14_gold_audit.log']:
    p = SRC/'outputs'/k
    if p.exists():
        shutil.copy(p, REPO/'metrics'/k); m += 1
print(f'metrics: {m} files')

# ---- app data: same schema the Dash app already expects, v8 vacancy swapped in ----
six = pd.read_csv(SIX)
v8 = pd.read_csv(SRC/'outputs'/'vacancy_scores_v8.csv')
d = six.merge(v8[['original_chip_id', 'vacancy_score', 'urban_percentage',
                  'water_percentage', 'other_percentage']],
              left_on='gid', right_on='original_chip_id', suffixes=('_v1', '_v8'))


def minmax(s):
    lo, hi = s.min(), s.max()
    return (s-lo)/(hi-lo)*100 if hi > lo else s*0


d['vacancy_subscore_v1'] = d['vacancy_subscore']          # keep the original for comparison
d['vacancy_subscore'] = minmax(d['vacancy_score_v8'])     # app.py reads this column
d['vacancy_score'] = d['vacancy_score_v8']
cols = ['gid', 'latitude', 'longitude', 'google_maps_link', 'vacancy_score',
        'vacancy_subscore', 'vacancy_subscore_v1', 'road_subscore',
        'healthcare_subscore', 'education_subscore', 'lifestyle_subscore',
        'essential_services_subscore', 'urban_percentage', 'water_percentage',
        'other_percentage', 'healthcare_count_1km', 'education_count_1km',
        'lifestyle_count_1km']
out = d[[c for c in cols if c in d.columns]]
out.to_csv(REPO/'data'/'6slider_parcel_scores.csv', index=False)
print(f'data: 6slider_parcel_scores.csv  {len(out)} parcels, v8 vacancy')

# developability flags from the audit
aud = SRC/'outputs'/'developability_audit.csv'
if aud.exists():
    a = pd.read_csv(aud)[['gid', 'river', 'inst', 'blocked']]
    a.to_csv(REPO/'data'/'developability_flags.csv', index=False)
    print(f'data: developability_flags.csv  {len(a)} parcels')

# ---- docs ----
for src_p, dst in [(SRC/'outputs'/'report_v6_built.html', 'audit_report.html'),
                   (SRC/'outputs'/'dashboard'/'parcel_ranker.html', 'parcel_ranker.html'),
                   (SRC/'outputs'/'label_fix_figure.png', 'label_fix_figure.png')]:
    if src_p.exists():
        shutil.copy(src_p, REPO/'docs'/dst)
print(f'docs: {len(list((REPO/"docs").iterdir()))} files')

(REPO/'.gitignore').write_text("""# raw rasters and chips -- 7.3 GB, regenerate with pipeline/01_chip.py and 08_chip64.py
*.npy
*.tif
# trained weights -- 54.8 MB per seed; see README for retraining
*.pt
# third-party imagery: Esri World Imagery is fetched for research, not redistributed
label_images/
_tiles/
__pycache__/
*.pyc
.ipynb_checkpoints/
""")
print('wrote .gitignore')

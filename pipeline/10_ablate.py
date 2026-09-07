"""Isolate the confound in the cumulative suite.

E1 (balanced sampling) and E2 (pretrained) each LOWERED macro-F1, yet E3/E4
carry them forward, so the context gain is measured on a handicapped base.
These runs strip them back out to find the genuinely best configuration.
"""
import json
from pathlib import Path
src = open('/home/prithvi/AcreVision_v2/scripts/09_upgrades.py').read()
exec(src[:src.index('R=[]')])          # reuse data prep + run() harness

O = Path('/home/prithvi/AcreVision_v2/outputs')
R = []
print('\n=== ISOLATION ABLATION ===', flush=True)
R.append(run('F1 custom+ctx+TTA',        X64, Y64, blk64, crop=True, tta=True))
R.append(run('F2 pretr+ctx+TTA (no bal)', X64, Y64, blk64, crop=True, tta=True, pre=True))
R.append(run('F3 custom+ctx+TTA 80ep',   X64, Y64, blk64, crop=True, tta=True, EP=80))
json.dump(R, open(O/'ablate.json','w'), indent=2)
print('\nsaved outputs/ablate.json', flush=True)

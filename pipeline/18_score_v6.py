"""City-wide vacancy scores from the v6 3-seed ensemble (64 px context + D4 TTA).

Improvements over 06_final4.py's scoring pass:
  * v6 labels (Vacant precision 68% on gold vs v4's 35%)
  * 3-seed ensemble, not a single seed
  * D4 test-time augmentation (optional: --tta; 8x compute, off by default on CPU)
  * 64 px context, cropped at each parcel's true offset so edge parcels are
    scored correctly instead of being silently mis-cropped

Per-seed normalisation is reconstructed by replaying the exact three-way
block-split RNG from 17_final_v6.py (train / val / test, gold excluded), so each
model is fed the statistics it was actually trained under.
"""
import numpy as np, torch, json, pandas as pd, time, os
import segmentation_models_pytorch as smp
from pathlib import Path

D = Path('/home/prithvi/AcreVision_v2/data'); O = Path('/home/prithvi/AcreVision_v2/outputs')
G = O/'gold'
import sys
LV = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else '6'
NSEED = 3
MODEL_TAG = 's' if LV == '8' else ''      # v8 models come from 27_train_v2.py (…v8s_seed…)
USE_TTA = ('--tta' in sys.argv)          # D4 TTA is 8x the compute; off by default on CPU
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu': torch.set_num_threads(int(os.environ.get('AV_THREADS', '32')))
eps = 1e-8; CLS = ['Vacant', 'Urban', 'Water', 'Other']
BM = {'blue': 1, 'green': 2, 'red': 3, 'nir': 7, 'swir1': 10}
print(f'labels=v{LV}  device={dev}  TTA={USE_TTA}  (pass --tta to enable; 8x slower)', flush=True)

valid = np.load(D/'valid.npy'); rc = np.load(D/'rowcol.npy'); gid = np.load(D/'gid.npy')
off = np.load(D/'off64.npy'); pbd = np.load(D/'parcel_building_density.npy')
lp = np.load(D/f'osm_label_frac{LV}.npy'); OSM = np.load(D/f'osm_labels{LV}.npy', mmap_mode='r')
CH64 = np.load(D/'chips64.npy', mmap_mode='r')

def feats_from(idx):
    x = np.asarray(CH64[idx], np.float32)
    r, g, b, nir, sw = [x[:, BM[k]] for k in ['red', 'green', 'blue', 'nir', 'swir1']]
    ndvi = (nir-r)/(nir+r+eps); ndwi = (nir-sw)/(nir+sw+eps); ndbi = (sw-nir)/(sw+nir+eps)
    return np.nan_to_num(np.concatenate([np.stack([r, g, b], 1)/3000.0,
                                         np.stack([ndvi, ndbi, ndwi], 1)], 1))

# ---- replay the training-fold statistics for each seed ----
tr_idx = np.where(valid & (lp >= 0.25))[0]
o = off[tr_idx]; tr_idx = tr_idx[(o[:, 0] == 16) & (o[:, 1] == 16)]
gold = set()
for gp in [G/'gold_claude.json', O/'gold2'/'gold_claude2.json', O/'gold3'/'gold_claude3.json']:
    if Path(gp).exists():
        gold |= {int(k) for k in json.load(open(gp)) if not k.startswith('_')}
gmask = np.array([g in gold for g in tr_idx])
def blocks(ii):
    r, c = rc[ii, 0], rc[ii, 1]; nb = 8
    return (np.digitize(r, np.quantile(r, np.linspace(0, 1, nb+1)[1:-1]))*nb
            + np.digitize(c, np.quantile(c, np.linspace(0, 1, nb+1)[1:-1])))
blk = blocks(tr_idx)
Xtr = feats_from(tr_idx)
norms = []
for seed in range(NSEED):
    ub = np.unique(blk); rng = np.random.default_rng(seed); rng.shuffle(ub)
    nte = max(1, int(.20*len(ub))); nva = max(1, int(.15*len(ub)))
    te_b, va_b = set(ub[:nte]), set(ub[nte:nte+nva])
    trn = np.array([(b not in te_b) and (b not in va_b) for b in blk]) & (~gmask)
    norms.append((Xtr[trn].mean((0, 2, 3), keepdims=True),
                  Xtr[trn].std((0, 2, 3), keepdims=True)+1e-6))
    print(f'  seed{seed} norm from {trn.sum()} chips', flush=True)
del Xtr

nets = []
for seed in range(NSEED):
    n = smp.Unet('resnet18', encoder_weights=None, in_channels=6, classes=4)
    n.load_state_dict(torch.load(O/f'model_v{LV}{MODEL_TAG}_seed{seed}.pt', map_location='cpu'))
    nets.append(n.to(dev).eval())
print(f'loaded {len(nets)} models', flush=True)

def tta_probs(net, x):
    acc = 0
    for k in range(4):
        xr = torch.rot90(x, k, (2, 3))
        acc = acc + torch.rot90(net(xr).softmax(1), -k, (2, 3))
        xf = torch.flip(xr, (3,))
        acc = acc + torch.rot90(torch.flip(net(xf).softmax(1), (3,)), -k, (2, 3))
    return acc/8

all_idx = np.where(valid)[0]
sc = np.zeros((len(all_idx), 4), np.float32)
t0 = time.time()
B = 128
for i in range(0, len(all_idx), B):
    ch = all_idx[i:i+B]
    Xb = feats_from(ch)
    prob = 0
    with torch.no_grad():
        for (mu, sd), net in zip(norms, nets):
            xb = torch.tensor(((Xb-mu)/sd).astype(np.float32)).to(dev)
            p_ = tta_probs(net, xb) if USE_TTA else net(xb).softmax(1)
            prob = prob + p_.cpu().numpy()
    prob /= len(nets)
    # crop each parcel at its TRUE 32px core offset within the 64px chip
    for j, g in enumerate(ch):
        r0, c0 = int(off[g, 0]), int(off[g, 1])
        p = prob[j][:, r0:r0+32, c0:c0+32].argmax(0)
        for k in range(4): sc[i+j, k] = (p == k).mean()*100
    if (i//B) % 25 == 0:
        el = time.time()-t0; done = i+len(ch)
        eta = el/done*(len(all_idx)-done)
        print(f'  {done}/{len(all_idx)}  [{el:.0f}s, ETA {eta/60:.1f}min]', flush=True)

df = pd.DataFrame({'original_chip_id': gid[all_idx], 'vacancy_score': sc[:, 0],
                   'urban_percentage': sc[:, 1], 'water_percentage': sc[:, 2],
                   'other_percentage': sc[:, 3], 'building_density': pbd[all_idx]})
df.to_csv(O/f'vacancy_scores_v{LV}.csv', index=False)
print(f'\nwrote {len(df)} parcels. mean vacancy {df.vacancy_score.mean():.2f}%', flush=True)

# ---- independent sanity check ----
res = {'n_parcels': int(len(df)), 'mean_vacancy': float(df.vacancy_score.mean())}
res['corr_v6_bdens'] = float(df.vacancy_score.corr(df.building_density))
print(f'\nINDEPENDENT CHECK -- corr(vacancy, OSM building density)')
print(f'  v{LV} (this run)                   : {res["corr_v6_bdens"]:+.3f}')
for tag, path, col in [('v2 (OSM-supervised, v4 labels)', O/'vacancy_scores_v2.csv', 'vacancy_score'),
                       ('v1 (leaky spectral-threshold)',
                        Path('/home/prithvi/osm_integration /vacancy_scores_valid_chips.csv'), 'vacancy_score')]:
    if not Path(path).exists(): continue
    old = pd.read_csv(path)
    m = df.merge(old[['original_chip_id', col]], on='original_chip_id', suffixes=('', '_o'))
    c = m[col+'_o'].corr(m.building_density) if col+'_o' in m else m[col].corr(m.building_density)
    k = 'corr_' + tag.split()[0] + '_bdens'
    res[k] = float(c)
    print(f'  {tag:32s}: {c:+.3f}   (n={len(m)})')
print('  (a valid vacancy score must be NEGATIVELY correlated with built-up density)')
json.dump(res, open(O/f'scores_v{LV}_summary.json', 'w'), indent=2)
print(f'saved outputs/vacancy_scores_v{LV}.csv', flush=True)

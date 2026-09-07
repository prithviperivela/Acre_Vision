"""Final model -- v2 of the training harness.

Three changes over 17_final_v6.py, all aimed at the instability that run exposed
(best epochs 11 / 4 / 28 on v6, and 13 / 10 / 9 on v4, with gold accuracy
swinging +-7.9 points on the old labels):

  1. LR SCHEDULE.  17_final used cosine annealing over a 60-epoch horizon that
     early stopping never reached, so runs ended while the learning rate was
     still near its maximum and validation was correspondingly noisy.  This uses
     ReduceLROnPlateau, which decays only when validation actually stalls, so the
     model is always annealed by the time stopping fires.
  2. SMOOTHED MODEL SELECTION.  Validation macro-F1 on ~1,500 chips is noisy
     enough that a single lucky epoch could win.  Selection now runs on a 3-epoch
     moving average, so the restored weights sit at a genuine plateau.
  3. FIVE SEEDS instead of three, and patience 10 instead of 8.

Evaluated on a held-out OSM spatial split and on BOTH verified gold batches --
batch 1 (used to fit the v7 road gate) and batch 2 (never used for any fitting,
so it is the honest test).

Config (E4, the winner once the cumulative confound was isolated):
  ImageNet-pretrained ResNet18-U-Net | 64 px input scored on the centre 32 px
  | balanced sampling | D4 augmentation | CE(class-weighted)+Dice | D4 TTA
  | early stopping on a spatially-separate validation split

Two things make this run different from 09_upgrades.py:
  1. Labels are v6 (scrub moved to Other, interstitial Vacant added) -- the
     answer key is 68% precise on Vacant instead of v4's 35%.
  2. The 96 gold chips are REMOVED from every training fold, so the gold score
     is a genuine held-out test against high-resolution-verified labels rather
     than OSM-vs-OSM agreement.
  3. Blocks are split three ways -- train / val / test -- all spatially disjoint.
     Training stops when validation macro-F1 has not improved for `PATIENCE`
     epochs and the best-validation weights are restored, so the reported test
     and gold numbers come from the model at its generalisation peak rather than
     from whatever the last epoch happened to produce.  The test blocks are
     never consulted for stopping, so this does not leak.

Usage: python 17_final_v6.py [label_version] [epochs] [seeds]
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, time, sys
import segmentation_models_pytorch as smp
from pathlib import Path
from collections import Counter

D = Path('/home/prithvi/AcreVision_v2/data'); O = Path('/home/prithvi/AcreVision_v2/outputs')
G = O/'gold'
LV  = sys.argv[1] if len(sys.argv) > 1 else '6'
EP  = int(sys.argv[2]) if len(sys.argv) > 2 else 60      # cap; ReduceLROnPlateau + early stopping end it
NSEED = int(sys.argv[3]) if len(sys.argv) > 3 else 5
PATIENCE = int(sys.argv[4]) if len(sys.argv) > 4 else 10  # epochs of no val improvement
SMOOTH = 3                                                # epochs in the selection average
EPOCH_CHIPS = 10000   # chips drawn per epoch. v8 carries 19.7k training chips, much of it
                      # near-duplicate rural farmland; capping the draw roughly halves epoch
                      # cost without reducing what the sampler can reach, since it draws with
                      # replacement from the full set every epoch.
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu': torch.set_num_threads(32)
eps = 1e-8; CLS = ['Vacant', 'Urban', 'Water', 'Other']
print(f'device={dev}  labels=v{LV}  epochs={EP}  seeds={NSEED}', flush=True)

valid = np.load(D/'valid.npy'); rc = np.load(D/'rowcol.npy')
OSM = np.load(D/f'osm_labels{LV}.npy', mmap_mode='r')
lp  = np.load(D/f'osm_label_frac{LV}.npy')
BM = {'blue': 1, 'green': 2, 'red': 3, 'nir': 7, 'swir1': 10}

def feats_from(arr, idx):
    x = np.asarray(arr[idx], np.float32)
    r, g, b, nir, sw = [x[:, BM[k]] for k in ['red', 'green', 'blue', 'nir', 'swir1']]
    ndvi = (nir-r)/(nir+r+eps); ndwi = (nir-sw)/(nir+sw+eps); ndbi = (sw-nir)/(sw+nir+eps)
    return np.nan_to_num(np.concatenate([np.stack([r, g, b], 1)/3000.0,
                                         np.stack([ndvi, ndbi, ndwi], 1)], 1))

def dice_loss(logit, y, ign=255):
    m = (y != ign)
    if m.sum() == 0: return logit.sum()*0
    p = logit.softmax(1); ys = torch.zeros_like(p)
    ys.scatter_(1, torch.where(m, y, torch.zeros_like(y)).unsqueeze(1), 1.0)
    mm = m.unsqueeze(1).float(); p = p*mm; ys = ys*mm
    inter = (p*ys).sum((0, 2, 3)); den = p.sum((0, 2, 3))+ys.sum((0, 2, 3))
    return 1-((2*inter+1)/(den+1)).mean()

def d4(x, y):
    k = np.random.randint(4); x = torch.rot90(x, k, (2, 3)); y = torch.rot90(y, k, (1, 2))
    if np.random.rand() < .5: x = torch.flip(x, (3,)); y = torch.flip(y, (2,))
    return x, y

def tta_probs(net, x):
    acc = 0
    for k in range(4):
        xr = torch.rot90(x, k, (2, 3))
        acc = acc + torch.rot90(net(xr).softmax(1), -k, (2, 3))
        xf = torch.flip(xr, (3,))
        acc = acc + torch.rot90(torch.flip(net(xf).softmax(1), (3,)), -k, (2, 3))
    return acc/8

def score(pt, yt):
    lab = yt != 255; iou = []; f1 = []
    for k in range(4):
        tp = int(((pt == k) & (yt == k) & lab).sum()); fp = int(((pt == k) & (yt != k) & lab).sum())
        fn = int(((pt != k) & (yt == k)).sum())
        iou.append(100*tp/max(tp+fp+fn, 1)); f1.append(100*2*tp/max(2*tp+fp+fn, 1))
    return dict(acc=100*float((pt[lab] == yt[lab]).mean()), macroF1=float(np.mean(f1)),
                iou={CLS[k]: round(iou[k], 1) for k in range(4)})

# ---------------- data ----------------
idx = np.where(valid & (lp >= 0.25))[0]
off = np.load(D/'off64.npy')[idx]; ok = (off[:, 0] == 16) & (off[:, 1] == 16)
idx = idx[ok]
CH64 = np.load(D/'chips64.npy', mmap_mode='r')
X = feats_from(CH64, idx); Y = np.asarray(OSM[idx], np.int64)
print(f'{len(idx)} chips with full 64px context', flush=True)

def load_gold(p):
    if not Path(p).exists(): return {}
    d = json.load(open(p))
    return {int(k): v for k, v in d.items() if not k.startswith('_')}
gold1 = load_gold(G/'gold_claude.json')
gold2 = load_gold(O/'gold2'/'gold_claude2.json')
gold3 = load_gold(O/'gold3'/'gold_claude3.json')
batches = {'gold1': gold1, 'gold2': gold2, 'gold3': gold3}
pos = {g: i for i, g in enumerate(idx)}
gmask = np.zeros(len(idx), bool)
allg = set(gold1) | set(gold2) | set(gold3)
for g in allg:
    if g in pos: gmask[pos[g]] = True
print(f'gold chips held out of training: {gmask.sum()} in-range of '
      f'{len(allg)} total (b1 {len(gold1)}, b2 {len(gold2)}, b3 {len(gold3)})', flush=True)

def blocks(ii):
    r, c = rc[ii, 0], rc[ii, 1]; nb = 8
    return (np.digitize(r, np.quantile(r, np.linspace(0, 1, nb+1)[1:-1]))*nb
            + np.digitize(c, np.quantile(c, np.linspace(0, 1, nb+1)[1:-1])))
blk = blocks(idx)
cnt = np.array([(Y == k).sum() for k in range(4)], float)
w_np = np.sqrt(cnt.sum()/(4*np.maximum(cnt, 1)))
print(f'class pixels {cnt.astype(int)} | weights {w_np.round(2)}', flush=True)

vac = (Y == 0).reshape(len(Y), -1).mean(1)
res_osm, res_gold, models, stop_info = [], [], [], []
t0 = time.time()
for seed in range(NSEED):
    torch.manual_seed(seed); np.random.seed(seed)
    ub = np.unique(blk); rng = np.random.default_rng(seed); rng.shuffle(ub)
    nte = max(1, int(.20*len(ub))); nva = max(1, int(.15*len(ub)))
    te_b, va_b = set(ub[:nte]), set(ub[nte:nte+nva])
    te  = np.array([b in te_b for b in blk])
    va  = np.array([b in va_b for b in blk]) & (~gmask)
    trn = np.array([(b not in te_b) and (b not in va_b) for b in blk]) & (~gmask)
    print(f'  seed{seed} blocks train/val/test = {trn.sum()}/{va.sum()}/{te.sum()} chips', flush=True)
    mu = X[trn].mean((0, 2, 3), keepdims=True); sd = X[trn].std((0, 2, 3), keepdims=True)+1e-6
    Xn = ((X-mu)/sd).astype(np.float32)
    xtr = torch.tensor(Xn[trn]); ytr = torch.tensor(Y[trn]); vtr = vac[trn]
    p_samp = np.where(vtr > 0, 1.0, 0.30); p_samp = p_samp/p_samp.sum()
    ckpt = O/f'model_v{LV}s_seed{seed}.pt'
    net = smp.Unet('resnet18', encoder_weights='imagenet', in_channels=6, classes=4).to(dev)
    resume = ckpt.exists()
    if resume:
        net.load_state_dict(torch.load(ckpt, map_location=dev))
        print(f'  seed{seed} RESUMED from {ckpt.name}; skipping training', flush=True)
    opt = torch.optim.Adam(net.parameters(), 5e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                     patience=3, min_lr=1e-5)
    w = torch.tensor(w_np, dtype=torch.float32, device=dev)
    n = len(xtr)
    xva = torch.tensor(Xn[va]); yva = Y[va]
    best_f1, best_state, best_ep, bad, hist = -1.0, None, 0, 0, []
    e = -1
    for e in range(EP if not resume else 0):
        net.train()
        order = np.random.choice(n, min(n, EPOCH_CHIPS), p=p_samp)
        for i in range(0, len(order), 64):
            j = order[i:i+64]
            xb, yb = xtr[j].to(dev), ytr[j].to(dev); xb, yb = d4(xb, yb)
            out = net(xb)[:, :, 16:48, 16:48]
            opt.zero_grad()
            (F.cross_entropy(out, yb, ignore_index=255, weight=w)+dice_loss(out, yb)).backward()
            opt.step()
        # ---- validation (plain forward, no TTA -- this is a stopping signal, not a report) ----
        net.eval(); pv = []
        with torch.no_grad():
            for i in range(0, len(xva), 128):
                pv.append(net(xva[i:i+128].to(dev))[:, :, 16:48, 16:48].argmax(1).cpu())
        vf1 = score(torch.cat(pv).numpy().reshape(-1), yva.reshape(-1))['macroF1']
        hist.append(round(vf1, 2))
        sch.step(vf1)                              # decay LR only when val stalls
        sm = float(np.mean(hist[-SMOOTH:]))        # select on a plateau, not a spike
        if sm > best_f1 + 1e-4:
            best_f1, best_ep, bad = sm, e+1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if (e+1) % 5 == 0 or bad == 0:
            lr = opt.param_groups[0]['lr']
            print(f'  seed{seed} ep{e+1}/{EP} valF1={vf1:.2f} sm={sm:.2f} '
                  f'best={best_f1:.2f}@{best_ep} bad={bad} lr={lr:.1e} '
                  f'[{time.time()-t0:.0f}s]', flush=True)
        if bad >= PATIENCE:
            print(f'  seed{seed} EARLY STOP at ep{e+1}; restoring ep{best_ep} '
                  f'(smoothed valF1={best_f1:.2f})', flush=True)
            break
    if best_state is not None: net.load_state_dict(best_state)
    torch.save(net.state_dict(), ckpt)          # persist BEFORE evaluating
    stop_info.append(dict(seed=seed, stopped_ep=e+1, best_ep=best_ep,
                          best_val_smoothed_macroF1=round(best_f1, 2), val_hist=hist))
    net.eval()

    def predict(sel):
        xs = torch.tensor(Xn[sel]); out = []
        with torch.no_grad():
            for i in range(0, len(xs), 64):
                p = tta_probs(net, xs[i:i+64].to(dev))[:, :, 16:48, 16:48]
                out.append(p.argmax(1).cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, 32, 32), np.int64)

    p_osm = predict(te)
    r1 = score(p_osm.reshape(-1), Y[te].reshape(-1)); res_osm.append(r1)

    # ---- gold evaluation, per batch (chip-level majority) ----
    print(f'  seed{seed}  OSM-holdout acc={r1["acc"]:.2f} F1={r1["macroF1"]:.2f} IoU={r1["iou"]}', flush=True)
    r2 = {}
    for bname, bd in batches.items():
        sc_ = {g: v for g, v in bd.items() if v != 'Mixed' and g in pos}
        if not sc_:
            print(f'  seed{seed}  {bname}: no in-range parcels, skipped', flush=True); continue
        keys = sorted(sc_)
        gi = np.array([pos[g] for g in keys]); gl = [sc_[g] for g in keys]
        p_g = predict(gi)
        pred_cls = [CLS[int(np.bincount(p.reshape(-1), minlength=4).argmax())] for p in p_g]
        gacc = 100*np.mean([a == b for a, b in zip(pred_cls, gl)])
        per = {}
        for c in CLS:
            tp = sum(1 for a, b in zip(pred_cls, gl) if a == c and b == c)
            fp = sum(1 for a, b in zip(pred_cls, gl) if a == c and b != c)
            fn = sum(1 for a, b in zip(pred_cls, gl) if a != c and b == c)
            per[c] = round(100*2*tp/max(2*tp+fp+fn, 1), 1)
        r2[bname] = dict(acc=gacc, f1=per, n=len(gl))
        print(f'  seed{seed}  {bname.upper()} n={len(gl)} acc={gacc:.1f}%  F1={per}', flush=True)
    res_gold.append(r2)
    json.dump(dict(osm=r1, gold=r2, stop=stop_info[-1]),
              open(O/f'seed_v{LV}s_{seed}.json', 'w'), indent=2)

def ms(vs): return float(np.mean(vs)), float(np.std(vs))
summary = dict(
    labels=f'v{LV}', device=dev, epoch_cap=EP, patience=PATIENCE, seeds=NSEED,
    n_chips=int(len(idx)), early_stop=stop_info,
    osm=dict(acc=ms([r['acc'] for r in res_osm]), macroF1=ms([r['macroF1'] for r in res_osm]),
             iou={c: ms([r['iou'][c] for r in res_osm]) for c in CLS}, per_seed=res_osm),
    gold={b: dict(acc=ms([r[b]['acc'] for r in res_gold if b in r]),
                  f1={c: ms([r[b]['f1'][c] for r in res_gold if b in r]) for c in CLS},
                  n=next(r[b]['n'] for r in res_gold if b in r))
          for b in batches if any(b in r for r in res_gold)},
    gold_per_seed=res_gold,
    sec=round(time.time()-t0))
json.dump(summary, open(O/f'final_v{LV}s.json', 'w'), indent=2)
print(f'\n=== FINAL (labels v{LV}) ===', flush=True)
print(f'  OSM held-out blocks : acc {summary["osm"]["acc"][0]:.2f}+-{summary["osm"]["acc"][1]:.2f}  '
      f'macroF1 {summary["osm"]["macroF1"][0]:.2f}+-{summary["osm"]["macroF1"][1]:.2f}', flush=True)
print('  IoU ' + ' '.join('%s=%.1f' % (c, summary['osm']['iou'][c][0]) for c in CLS), flush=True)
for b, gg in summary['gold'].items():
    print('  %-6s (n=%d, never trained on): acc %.1f+-%.1f%%   Vacant F1 %.1f+-%.1f'
          % (b.upper(), gg['n'], gg['acc'][0], gg['acc'][1],
             gg['f1']['Vacant'][0], gg['f1']['Vacant'][1]), flush=True)
for b, gg in summary['gold'].items():
    print('  %-6s F1 ' % b.upper() + ' '.join('%s=%.1f' % (c, gg['f1'][c][0]) for c in CLS), flush=True)
for si in stop_info:
    print(f'  seed{si["seed"]}: stopped ep{si["stopped_ep"]}, best ep{si["best_ep"]} '
          f'(smoothed val macroF1 {si["best_val_smoothed_macroF1"]})', flush=True)
print(f'saved outputs/final_v{LV}s.json  [{summary["sec"]}s]', flush=True)

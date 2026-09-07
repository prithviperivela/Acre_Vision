# Acre Vision

Vacant-land detection for Hyderabad from Sentinel-2, with an OpenStreetMap-derived
suitability ranker on top.

**~84% accuracy** against high-resolution verified ground truth, city-weighted.
29,951 parcels of 25.3 acres each, across a 3,090 km² area of interest.

---

## The one rule this project rests on

**Labels come only from OpenStreetMap vector geometry. The network sees only
Sentinel-2 pixels.** Neither can see the other's source.

The first version of this project reported 98.5% accuracy. That number was an
artefact: its labels were computed from the same spectral indices (NDVI, NDBI,
NDWI) that were fed to the model as input, so the model was being graded on its
ability to reproduce a rule it had already been handed. Everything here follows
from removing that circularity.

Do not reintroduce a spectral term into any labelling rule, however convenient.

---

## Results

Answer-key accuracy, weighted to the true composition of the city:

| labels | on OSM-labelled 40% | on unlabelled 60% | city-weighted |
|---|---|---|---|
| v4 | 50.0% | 4.2% | 22.6% |
| v6 | 65.8% | 8.3% | 31.4% |
| **v8** | 65.8% | **91.7%** | **81.3%** |

Model, both ensembled over 3 seeds and run over the **same** 256 verified parcels:

| trained on | batch 1 (urban) | batch 2 (mixed) | batch 3 (rural) | all 256 | Vacant F1 |
|---|---|---|---|---|---|
| v6 labels | 72.9% | 72.0% | 65.0% | 70.7% | 74.3 |
| **v8 labels** | 72.9% | **76.0%** | **91.7%** | **78.5%** | **83.3** |

Independent check needing no labels — a vacancy score must fall as built-up
density rises:

| version | corr(vacancy, OSM building density) |
|---|---|
| v1 (original spectral threshold) | −0.085 |
| v2 | −0.379 |
| v6 | −0.464 |
| **v8** | **−0.651** |

---

## Repository layout

```
app.py                     Dash suitability dashboard (deployed on Render)
data/
  6slider_parcel_scores.csv    29,532 parcels; vacancy_subscore is v8,
                               vacancy_subscore_v1 kept for comparison
  developability_flags.csv     river-corridor / institutional flags
docs/
  project_report.html          full project report
  audit_report.html            the label-audit write-up
  parcel_ranker.html           standalone dashboard, no server needed
pipeline/                  the whole pipeline, numbered in run order
labels/                    the 256 verified parcels + sampling records
metrics/                   every result JSON quoted in the report
```

## Pipeline

| script | does |
|---|---|
| `01_chip.py`, `08_chip64.py` | tile the raster into 32 px parcels, and a 64 px context version |
| `02b/02c/02d/02f_osm_labels*.py` | label versions v4 → v8 |
| `12/25/30_fetch_*.py` | fetch Esri imagery for the verified batches |
| `13/36_*.py` | render blind labelling sheets |
| `14/28/32_*.py` | score label rules against verified ground |
| `15_fclass_blame.py` | attribute label error to individual OSM tags |
| **`27_train_v2.py`** | **trainer.** `python pipeline/27_train_v2.py 8 40 3 10` |
| **`18_score_v6.py`** | **city-wide scoring.** `python pipeline/18_score_v6.py 8` |
| `35_fair_model_compare.py` | run two models over the *same* parcels — use this, not per-run numbers |
| `37_label_gold.py`, `38_agreement.py` | human labelling and Fleiss' kappa |
| `40/41/42_*.py` | dashboard data, developability audit, repo build |

### Reproducing

Raw chips (7.3 GB) and trained weights (54.8 MB per seed) are **not** in this
repository. Regenerate with:

```bash
python pipeline/01_chip.py && python pipeline/08_chip64.py   # chips
python pipeline/02f_osm_labels8.py                           # labels
python pipeline/27_train_v2.py 8 40 3 10                     # train (~1h GPU)
python pipeline/18_score_v6.py 8                             # city-wide scores
```

Esri World Imagery tiles are fetched for research by scripts 12/25/30/36 and are
not redistributed here.

---

## The verified ("gold") sets

256 parcels classified from Esri World Imagery at zoom 18 (~0.57 m/px), **blind
to the OSM mask** and shuffled.

| set | n | purpose |
|---|---|---|
| `labels/gold_claude_batch1.json` | 96 | diagnostic; chose the `scrub` fix and fitted the v7 gate, so optimistic for both |
| `labels/gold_claude_batch2.json` | 100 | clean test; added the previously unmeasured *unlabelled* stratum |
| `labels/gold_claude_batch3.json` | 60 | drawn only from land v8 newly claims; measures the new rule directly |

**Do not tune anything on a batch you then quote as its test.** That rule is what
caught v7: it looked like a 68→84% precision win on the batch it was fitted to,
and did not transfer.

> **These labels are annotated by a model, not by humans.** Read every accuracy
> figure as "agreement with one careful reader" until `37_label_gold.py` has been
> run by three people and `38_agreement.py` reports the kappa.

---

## Known failure modes

Measured:

| mode | share of city | share of top 25 |
|---|---|---|
| River / stream corridor (Musi floodplain) | 7.4% | **16.0%** |
| Institutional curtilage (campus grounds) | 5.5% | **8.0%** |

Both are open land, correctly classified, and not acquirable. They concentrate at
the top of the ranking because they are simultaneously open *and* central, which
is exactly what the composite rewards. `data/developability_flags.csv` filters
them; the filter is on by default in `app.py`.

Reasoned but untested — dry-season bias (imagery is Jan–Mar; "vacant" may be a
seasonal state rather than a property of the land), active construction sites,
quarries and landfill, Deccan rocky outcrop, temporal staleness, mixed-pixel
boundaries, and amenity subscores that systematically penalise rural land. See
`docs/project_report.html` for the full treatment.

**Parcel size is not a design choice.** 32 px × 10 m GSD = a 320 m square =
25.3 acres. Both measured failure modes are partly artefacts of that granularity;
at 0.5 m GSD a 32 px parcel is 0.06 acres and a campus stops sharing a chip with
the plot next door.

---

## Deployment

```bash
pip install -r requirements.txt
python app.py
```

`render.yaml` deploys it as-is. The vacancy subscore served by the app is the v8
model; the developability filter defaults to on.

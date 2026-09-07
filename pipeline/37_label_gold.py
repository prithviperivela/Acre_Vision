"""Human labelling of the 256 verified parcels.  Usage:

    python 37_label_gold.py <your_name>

All three annotators label ALL 256 in the SAME fixed order, independently.
Do not split the set between you and do not confer -- the whole point is to
measure whether three people, working alone, arrive at the same answer. If they
do, the class definitions are sound and the ground truth is trustworthy. If they
do not, we learn exactly which parcels are ambiguous, which is also worth knowing.

Keys:  1 Vacant   2 Urban   3 Water   4 Other   5 Mixed/unsure
       LEFT go back and change the previous answer      q  save and quit

Progress saves after every keypress, so you can stop and resume any time.

Class definitions -- apply to the CENTRE-DOMINANT land in the 320 m square:
  Vacant  open developable ground: bare earth, scrub, grass, farmland, orchard,
          plotted-but-empty land. The thing we are trying to find.
  Urban   built structures plus their immediate curtilage cover >= half the square.
  Water   open water dominant.
  Other   dominated by something real but NOT developable: dense forest or woodland,
          park, quarry, airfield, sewage works, solar farm.
  Mixed   genuinely undecidable, or the image is broken. Excluded from scoring --
          use it, do not agonise, but do not use it to avoid a hard call.
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

O = Path('/home/prithvi/AcreVision_v2/outputs'); IM = O/'label_images'
who = (sys.argv[1] if len(sys.argv) > 1 else 'anon').lower()
CLASSES = {'1': 'Vacant', '2': 'Urban', '3': 'Water', '4': 'Other', '5': 'Mixed'}
order = json.load(open(IM/'order.json'))
out = O/f'gold_human_{who}.json'
done = json.load(open(out)) if out.exists() else {}
print(f'{who}: {len(done)}/{len(order)} already labelled -> {out}')

state = {'i': 0}
while state['i'] < len(order) and str(order[state['i']]['gid']) in done:
    state['i'] += 1

fig, ax = plt.subplots(figsize=(8.4, 9.0))
try:
    fig.canvas.manager.set_window_title(f'Acre Vision labelling - {who}')
except Exception:
    pass


def draw():
    i = state['i']
    ax.clear(); ax.axis('off')
    if i >= len(order):
        ax.set_title(f'ALL {len(order)} DONE - thank you.\nclose this window',
                     fontsize=15, fontweight='bold')
        fig.canvas.draw(); return
    s = order[i]
    ax.imshow(mpimg.imread(IM/f'{s["gid"]}.jpg'))
    prev = done.get(str(s['gid']), '')
    ax.set_title(f'{i+1} / {len(order)}      gid {s["gid"]}'
                 f'{"      [currently: " + prev + "]" if prev else ""}\n'
                 '1 Vacant    2 Urban    3 Water    4 Other    5 Mixed\n'
                 '320 m across        LEFT = back        q = save & quit',
                 fontsize=12)
    fig.canvas.draw()


def onkey(e):
    if e.key == 'q':
        plt.close(fig); return
    if e.key == 'left':
        state['i'] = max(0, state['i'] - 1); draw(); return
    if e.key in CLASSES and state['i'] < len(order):
        done[str(order[state['i']]['gid'])] = CLASSES[e.key]
        json.dump(done, open(out, 'w'), indent=1)
        state['i'] += 1; draw()


fig.canvas.mpl_connect('key_press_event', onkey)
draw(); plt.show()
print(f'saved {len(done)} labels -> {out}')
if len(done) < len(order):
    print(f'{len(order) - len(done)} left. Re-run the same command to carry on.')

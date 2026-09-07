"""Gold-standard labelling tool. Three annotators label disjoint chip sets.
Chip-level majority class (not per-pixel) -- fast, ~4s/chip, ~85 chips each.
Usage:  python 04_label_tool.py <annotator_name>"""
import numpy as np, json, sys
from pathlib import Path
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

D=Path('/home/prithvi/AcreVision_v2/data'); O=Path('/home/prithvi/AcreVision_v2/outputs')
who=sys.argv[1] if len(sys.argv)>1 else 'anon'
N_PER=85; CLASSES={'1':'Vacant','2':'Urban','3':'Water','4':'Mixed/Unclear'}

CH=np.load(D/'chips.npy',mmap_mode='r'); valid=np.load(D/'valid.npy')
gid=np.load(D/'gid.npy')
# fixed shared sample -> disjoint slice per annotator, reproducible
rng=np.random.default_rng(42); pool=np.where(valid)[0]
sample=rng.choice(pool,255,replace=False)
who_i={'prithvi':0,'shashank':1,'anuraga':2}.get(who.lower(),0)
mine=sample[who_i*N_PER:(who_i+1)*N_PER]

out=O/f'gold_{who}.json'; done=json.load(open(out)) if out.exists() else {}
print(f'{who}: {len(done)}/{len(mine)} already labelled. Keys 1-4, q=quit.')

def show(i):
    x=np.asarray(CH[i],np.float32)
    rgb=np.stack([x[3],x[2],x[1]],-1)                 # R,G,B
    rgb=np.clip(rgb/np.nanpercentile(rgb,98),0,1)
    return np.nan_to_num(rgb)

fig,ax=plt.subplots(figsize=(6,6.6)); state={'k':0}
def draw():
    todo=[i for i in mine if str(i) not in done]
    if not todo:
        ax.clear(); ax.set_title('ALL DONE - close window'); fig.canvas.draw(); return
    i=todo[0]; state['cur']=i
    ax.clear(); ax.imshow(show(i),interpolation='nearest'); ax.axis('off')
    ax.set_title(f'{who}  {len(done)}/{len(mine)}   gid={gid[i]}\n'
                 '1=Vacant  2=Urban  3=Water  4=Mixed/Unclear',fontsize=11)
    fig.canvas.draw()
def onkey(e):
    if e.key=='q': plt.close(fig); return
    if e.key in CLASSES:
        done[str(state['cur'])]=CLASSES[e.key]
        json.dump(done,open(out,'w'),indent=1); draw()
fig.canvas.mpl_connect('key_press_event',onkey); draw(); plt.show()
print(f'saved {len(done)} labels -> {out}')

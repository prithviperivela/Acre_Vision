"""Assemble the part-two report: CSS + body + measured numbers + embedded figures."""
import json, base64, datetime
from pathlib import Path
O=Path('/home/prithvi/AcreVision_v2/outputs')
css=(O/'_report_css.html').read_text(); body=(O/'_report_body.html').read_text()
v8=json.load(open(O/'final_v8s.json'))
sc=json.load(open(O/'scores_v8_summary.json')) if (O/'scores_v8_summary.json').exists() else {}
def b64(p): return 'data:image/jpeg;base64,'+base64.b64encode(Path(p).read_bytes()).decode()
g=v8['gold']
vals={
 '@@DATE@@'    : datetime.date.today().strftime('%-d %B %Y'),
 '@@V8G1@@'    : f"{g['gold1']['acc'][0]:.1f}" if 'gold1' in g else 'n/a',
 '@@V8G2@@'    : f"{g['gold2']['acc'][0]:.1f}" if 'gold2' in g else 'n/a',
 '@@V8G3@@'    : f"{g['gold3']['acc'][0]:.1f}" if 'gold3' in g else 'n/a',
 '@@V8G2VAC@@' : f"{g['gold2']['f1']['Vacant'][0]:.1f}" if 'gold2' in g else 'n/a',
 '@@V8OSMF1@@' : f"{v8['osm']['macroF1'][0]:.1f}",
 '@@CORR8@@'   : f"{sc.get('corr_v6_bdens', float('nan')):+.3f}",
 '@@MEANVAC@@' : f"{sc.get('mean_vacancy', float('nan')):.1f}",
 '@@FIG_FIX@@' : b64(O/'fig_labelfix_web.jpg'),
}
s=css+body
for k,v in vals.items(): s=s.replace(k,v)
out=O/'report_v6_built.html'          # same path -> same artifact URL
out.write_text(s)
import re
left=sorted(set(re.findall(r'@@[A-Z0-9_]+@@',s)))
print(f'wrote {out} ({len(s)/1024:.0f} KB); unresolved tokens: {left or "none"}')
for k,v in vals.items():
    if not k.startswith('@@FIG'): print(f'  {k:14s} = {v}')

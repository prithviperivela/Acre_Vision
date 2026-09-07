"""Inject measured numbers and base64 figures into the report template."""
import json, base64, datetime
from pathlib import Path
O = Path('/home/prithvi/AcreVision_v2/outputs')

v6 = json.load(open(O/'final_v6.json'))
v4 = json.load(open(O/'final_v4.json'))

def b64(p, mime='image/jpeg'):
    return f'data:{mime};base64,' + base64.b64encode(Path(p).read_bytes()).decode()

vals = {
 '@@DATE@@'   : datetime.date.today().strftime('%-d %B %Y'),
 '@@V6GACC@@' : f"{v6['gold']['acc'][0]:.1f}",
 '@@V4GACC@@' : f"{v4['gold']['acc'][0]:.1f}",
 '@@V6GVAC@@' : f"{v6['gold']['f1']['Vacant'][0]:.1f}",
 '@@V4GVAC@@' : f"{v4['gold']['f1']['Vacant'][0]:.1f}",
 '@@V6IOU@@'  : f"{v6['osm']['iou']['Vacant'][0]:.1f}",
 '@@V4IOU@@'  : f"{v4['osm']['iou']['Vacant'][0]:.1f}",
 '@@V6F1@@'   : f"{v6['osm']['macroF1'][0]:.1f}",
 '@@V4F1@@'   : f"{v4['osm']['macroF1'][0]:.1f}",
 '@@V6OTH@@'  : f"{v6['gold']['f1']['Other'][0]:.1f}",
 '@@FIG_GOLD@@': b64(O/'fig_gold_web.jpg'),
 '@@FIG_FIX@@' : b64(O/'fig_labelfix_web.jpg'),
}
s = (O/'report_v6.html').read_text()
for k, v in vals.items():
    if k not in s and not k.startswith('@@FIG'): print(f'  WARN token {k} not found')
    s = s.replace(k, v)
left = [t for t in s.split('@@') if len(t) < 12 and t.isupper() and t]
out = O/'report_v6_built.html'
out.write_text(s)
print(f'wrote {out}  ({len(s)/1024:.0f} KB)')
for k, v in vals.items():
    if not k.startswith('@@FIG'): print(f'  {k:12s} = {v}')

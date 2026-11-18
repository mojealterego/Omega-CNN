import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from engine import Engine
from conformal import Conformal
x,y=load_digits(return_X_y=True); x=x.reshape(-1,8,8)
a,z=train_test_split(np.arange(len(y)),test_size=.6,random_state=42,stratify=y); b,c=train_test_split(z,test_size=.5,random_state=43,stratify=y[z]); m=Conformal(Engine('digits_cnn_trained.onnx')).fit(x[a],x[b],y[b]); rng=np.random.default_rng(42)
def metrics(x,y):
 r=m.predict(x); keep=~r['rejected']; return dict(n=len(y),set_coverage=float(r['sets'][np.arange(len(y)),y].mean()),acceptance=float(keep.mean()),accepted_accuracy=float((r['labels'][keep]==y[keep]).mean()) if keep.any() else None,ood_rate=float(r['ood'].mean()),mean_set_size=float(r['sets'].sum(1).mean()))
print(json.dumps(dict(reference_n=len(a),calibration_n=len(b),q=m.q,distance_threshold=m.d,clean=metrics(x[c],y[c]),noise=metrics(np.clip(x[c]+rng.normal(0,2,x[c].shape),0,16),y[c]),uniform_ood=float(m.predict(rng.uniform(0,16,(1000,8,8)))['ood'].mean()),blank_ood=float(m.predict(np.zeros((100,8,8)))['ood'].mean()),training_overlap_unknown=True),indent=2))

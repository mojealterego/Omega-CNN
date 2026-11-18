import json,time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from engine import Engine
from adaptive import Adaptive,calibrate
x,y=load_digits(return_X_y=True); x=x.reshape(-1,8,8)
a,b=train_test_split(np.arange(len(y)),test_size=.5,random_state=42,stratify=y)
e=Engine('digits_cnn_trained.onnx'); base=Adaptive(e,reject_threshold=0)
t=calibrate(base.predict(x[a])['probabilities'],y[a]); model=Adaptive(e,temperature=t)
def metrics(x,y):
 start=time.perf_counter(); r=model.predict(x); elapsed=time.perf_counter()-start
 keep=~r['rejected']; p=r['probabilities']; return dict(n=len(y),coverage=float(keep.mean()),accuracy_all=float((r['labels']==y).mean()),accuracy_accepted=float((r['labels'][keep]==y[keep]).mean()) if keep.any() else None,escalation=float(r['escalated'].mean()),nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-30)).mean()),seconds=elapsed)
rng=np.random.default_rng(42)
print(json.dumps({'temperature':t,'calibration_n':len(a),'evaluation_n':len(b),'clean':metrics(x[b],y[b]),'noise_sigma_2':metrics(np.clip(x[b]+rng.normal(0,2,x[b].shape),0,16),y[b]),'blank_rejection':float(model.predict(np.zeros((100,8,8)))['rejected'].mean()),'uniform_noise_rejection':float(model.predict(rng.uniform(0,16,(1000,8,8)))['rejected'].mean()),'training_overlap_unknown':True},indent=2))

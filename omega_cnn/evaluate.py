import json,time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from engine import Engine
x,y=load_digits(return_X_y=True);x=x.reshape(-1,8,8);e=Engine('digits_cnn_trained.onnx')
r={}
for mode in (False,True):
 e.predict(x[:64],tta=mode); times=[]
 for _ in range(5):
  t=time.perf_counter();p=e.predict(x,tta=mode);times.append(time.perf_counter()-t)
 pred=p.argmax(1);r[str(mode)]={'accuracy':float((pred==y).mean()),'errors':int((pred!=y).sum()),'median_seconds':float(np.median(times)),'confusion':confusion_matrix(y,pred).tolist()}
print(json.dumps(r,indent=2))

"""Selective inference with confidence routing and temperature scaling."""
import numpy as np
from scipy.optimize import minimize_scalar

def rescale(p, temperature):
 z=np.log(np.maximum(p,1e-30))/temperature
 z-=z.max(axis=1,keepdims=True)
 q=np.exp(z); return q/q.sum(axis=1,keepdims=True)

def calibrate(p,y):
 p=np.asarray(p,dtype=np.float64); y=np.asarray(y)
 if p.ndim!=2 or not len(p) or y.shape!=(len(p),) or y.dtype.kind not in 'iu' or (y<0).any() or (y>=p.shape[1]).any() or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1): raise ValueError('Invalid calibration data')
 def loss(logt): return -np.log(np.maximum(rescale(p,np.exp(logt))[np.arange(len(y)),y],1e-30)).mean()
 r=minimize_scalar(loss,bounds=(-3,3),method='bounded'); return float(np.exp(r.x))

class Adaptive:
 def __init__(self,engine,*,threshold=.95,reject_threshold=.8,temperature=1.,min_foreground=.01):
  if not np.isfinite([threshold,reject_threshold,temperature,min_foreground]).all() or not 0<=threshold<=1 or not 0<=reject_threshold<=1 or not 0<=min_foreground<=1 or not .01<=temperature<=100: raise ValueError('Invalid configuration')
  self.engine=engine; self.threshold=threshold; self.reject_threshold=reject_threshold; self.temperature=temperature; self.min_foreground=min_foreground
 def predict(self,images,*,scale=16,batch_size=64):
  x=np.asarray(images); p=self.engine.predict(x,scale=scale,batch_size=batch_size)
  blank=(x.max(axis=(1,2))-x.min(axis=(1,2)))<=self.min_foreground*scale
  route=(p.max(1)<=self.threshold)&~blank
  if route.any(): p[route]=self.engine.predict(x[route],scale=scale,batch_size=batch_size,tta=True)
  p=rescale(p,self.temperature); confidence=p.max(1); rejected=blank|(confidence<self.reject_threshold)
  return {'probabilities':p,'confidence':confidence,'labels':np.where(rejected,-1,p.argmax(1)),'rejected':rejected,'escalated':route}

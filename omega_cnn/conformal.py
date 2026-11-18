"""Split conformal class sets and independent pixel-space novelty gate."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from adaptive import Adaptive

def quantile(scores,alpha):
 s=np.asarray(scores,dtype=float)
 if s.ndim!=1 or not len(s) or not np.isfinite(s).all() or not np.isfinite(alpha) or not 0<alpha<1: raise ValueError('Invalid scores or alpha')
 k=int(np.ceil((len(s)+1)*(1-alpha)))
 return float(np.partition(s,k-1)[k-1]) if k<=len(s) else float('inf')

class Conformal:
 def __init__(self,engine,alpha=.01,ood_alpha=.025):
  quantile([0],alpha); quantile([0],ood_alpha)
  self.alpha=alpha; self.ood_alpha=ood_alpha; self.model=Adaptive(engine,reject_threshold=0); self.nn=None
 def _features(self,x):
  x=np.asarray(x)
  if x.ndim!=3 or x.shape[1:]!=(8,8) or not 1<=len(x)<=100000 or x.dtype.kind not in 'uif' or not np.isfinite(x).all() or x.min()<0 or x.max()>16: raise ValueError('Expected finite N x 8 x 8 pixels in [0,16]')
  return x.astype(np.float64).reshape(-1,64)/16
 def _distance(self,features):
  return np.concatenate([self.nn.kneighbors(features[i:i+256],return_distance=True)[0][:,-1] for i in range(0,len(features),256)])
 def fit(self,reference,calibration,y):
  a=self._features(reference); b=self._features(calibration); y=np.asarray(y)
  if len(a)<5 or y.shape!=(len(b),) or y.dtype.kind not in 'iu' or (y<0).any() or (y>9).any(): raise ValueError('Invalid reference or labels')
  nn=NearestNeighbors(n_neighbors=5,algorithm='brute').fit(a)
  p=self.model.predict(calibration)['probabilities']; q=quantile(1-p[np.arange(len(y)),y],self.alpha)
  self.nn=nn; self.q=q; self.d=quantile(self._distance(b),self.ood_alpha); return self
 def predict(self,x):
  if self.nn is None: raise RuntimeError('Fit required')
  f=self._features(x); p=self.model.predict(x)['probabilities']; sets=1-p<=self.q
  distance=self._distance(f); ood=(distance>self.d)|(np.ptp(f,axis=1)<=.01)
  rejected=ood|(sets.sum(1)!=1)
  return dict(probabilities=p,sets=sets,ood=ood,distance=distance,rejected=rejected,labels=np.where(rejected,-1,sets.argmax(1)))

"""Bounded, thread-safe OpenCV inference with optional translation ensemble."""
import threading
import cv2
import numpy as np
class Engine:
 def __init__(self,path):
  self.net=cv2.dnn.readNetFromONNX(str(path)); self.lock=threading.Lock()
 def predict(self,images,*,scale=16,batch_size=64,tta=False):
  x=np.asarray(images)
  if x.ndim!=3 or x.shape[1:]!=(8,8) or not 0<len(x)<=100000: raise ValueError('Expected N x 8 x 8, 1..100000 images')
  if scale not in (1,16,255) or not isinstance(batch_size,int) or not 1<=batch_size<=256: raise ValueError('Invalid scale or batch size')
  if x.dtype.kind not in 'uif' or not np.isfinite(x).all() or x.min()<0 or x.max()>scale: raise ValueError('Invalid pixel values')
  x=x.astype(np.float32)/scale; results=[]
  for start in range(0,len(x),batch_size):
   b=x[start:start+batch_size]; variants=[b]
   if tta:
    padded=np.pad(b,((0,0),(1,1),(1,1)))
    variants.extend(padded[:,1+dy:9+dy,1+dx:9+dx] for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)])
   ps=[]
   with self.lock:
    for v in variants:
     self.net.setInput(np.ascontiguousarray(v[:,None])); ps.append(self.net.forward().copy())
   p=ps[0] if not tta else .6*ps[0]+.1*sum(ps[1:])
   if p.shape!=(len(b),10) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5): raise RuntimeError('Invalid model output')
   results.append(p)
  return np.concatenate(results)

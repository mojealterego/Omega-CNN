import unittest
import numpy as np
from conformal import quantile, Conformal
from engine import Engine
from sklearn.datasets import load_digits
class Tests(unittest.TestCase):
 def test_quantile(self):
  self.assertEqual(quantile(np.arange(9),.2),7)
  self.assertTrue(np.isinf(quantile(np.arange(9),.01)))
 def test_invalid(self):
  for a in [0,1,np.nan]:
   with self.assertRaises(ValueError): quantile([1],a)
 def test_real(self):
  x,y=load_digits(return_X_y=True); x=x.reshape(-1,8,8)
  m=Conformal(Engine('digits_cnn_trained.onnx')).fit(x[:500],x[500:1000],y[500:1000])
  r=m.predict(x[1000:1100]); self.assertEqual(r['sets'].shape,(100,10)); self.assertTrue((r['labels'][r['sets'].sum(1)!=1]==-1).all())
  self.assertTrue(m.predict(np.zeros((2,8,8)))['rejected'].all())
  with self.assertRaises(ValueError): m.predict(np.full((1,8,8),np.nan))

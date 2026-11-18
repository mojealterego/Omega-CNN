import unittest
import numpy as np
from engine import Engine
from adaptive import Adaptive, calibrate
class AdaptiveTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.e=Engine('digits_cnn_trained.onnx')
 def test_temperature(self):
  p=np.array([[.9,.1],[.2,.8]])
  t=calibrate(p,np.array([0,1])); self.assertGreater(t,0); self.assertLess(t,1)
 def test_blank(self):
  r=Adaptive(self.e).predict(np.zeros((2,8,8))); self.assertTrue(r['rejected'].all())
 def test_cascade(self):
  x=np.random.default_rng(42).uniform(0,16,(10,8,8)); r=Adaptive(self.e,threshold=1).predict(x); self.assertTrue(r['escalated'].all()); np.testing.assert_allclose(r['probabilities'],self.e.predict(x,tta=True),atol=1e-6)
 def test_invalid_config(self):
  for v in [0,-1,float('nan')]:
   with self.assertRaises(ValueError): Adaptive(self.e,temperature=v)
 def test_invalid_input(self):
  with self.assertRaises(ValueError): Adaptive(self.e).predict(np.zeros((1,9,9)))

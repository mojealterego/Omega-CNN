import unittest
import numpy as np
from engine import Engine
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.e=Engine('digits_cnn_trained.onnx')
 def test_batch(self):
  p=self.e.predict(np.zeros((3,8,8))); self.assertEqual(p.shape,(3,10)); np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
 def test_invalid(self):
  for x in [np.zeros((1,28,28)),np.full((1,8,8),np.nan),np.full((1,8,8),17),np.empty((0,8,8))]:
   with self.assertRaises(ValueError): self.e.predict(x)
 def test_scale(self):
  x=np.ones((2,8,8)); np.testing.assert_allclose(self.e.predict(x*16),self.e.predict(x,scale=1))
 def test_chunk(self):
  x=np.random.default_rng(42).uniform(0,16,(5,8,8)); np.testing.assert_allclose(self.e.predict(x,batch_size=1),self.e.predict(x,batch_size=5),atol=1e-6)
 def test_translation(self):
  x=np.zeros((2,8,8)); p=self.e.predict(x,tta=True); np.testing.assert_allclose(p.sum(1),1,atol=1e-6)

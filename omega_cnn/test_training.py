import unittest
import torch
from training import CNN, pgd
class TrainingTests(unittest.TestCase):
 def test_shape(self):
  for w in (16,32): self.assertEqual(CNN(w)(torch.zeros(3,1,8,8)).shape,(3,10))
 def test_attack_bounds_and_no_parameter_grads(self):
  m=CNN(16); x=torch.rand(4,1,8,8); y=torch.arange(4); z=pgd(m,x,y,.12,3)
  self.assertTrue((z-x).abs().max()<=.120001); self.assertTrue(z.min()>=0 and z.max()<=1); self.assertTrue(all(p.grad is None for p in m.parameters()))
 def test_attack_increases_loss(self):
  torch.manual_seed(5); m=CNN(16); x=torch.rand(4,1,8,8); y=torch.arange(4); z=pgd(m,x,y,.12,3)
  self.assertGreaterEqual(torch.nn.functional.cross_entropy(m(z),y).item(),torch.nn.functional.cross_entropy(m(x),y).item())

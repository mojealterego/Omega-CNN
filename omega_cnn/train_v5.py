import json, torch, numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from training import CNN, pgd
import torch.nn.functional as F

def main():
 torch.set_num_threads(2); torch.manual_seed(42); torch.use_deterministic_algorithms(True)
 x,y=load_digits(return_X_y=True); x=torch.tensor(x.reshape(-1,1,8,8)/16,dtype=torch.float32); y=torch.tensor(y)
 a,c=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42); a,b=train_test_split(a,test_size=.25,stratify=y[a],random_state=43)
 def evaluate(m,i):
  z=pgd(m,x[i],y[i],.12,10)
  with torch.no_grad(): return {'clean':float((m(x[i]).argmax(1)==y[i]).float().mean()),'pgd':float((m(z).argmax(1)==y[i]).float().mean())}
 results=[]; best=-1
 for w in (16,32):
  for robust in (False,True):
   torch.manual_seed(42); m=CNN(w); opt=torch.optim.AdamW(m.parameters(),lr=.002)
   for epoch in range(20):
    for i in torch.tensor(a)[torch.randperm(len(a))].split(64):
     v=x[i]; t=y[i]; z=pgd(m,v,t) if robust else None; opt.zero_grad(); loss=F.cross_entropy(m(v),t)
     if robust: loss=.5*(loss+F.cross_entropy(m(z),t))
     loss.backward(); opt.step()
   val=evaluate(m,b); score=.5*(val['clean']+val['pgd']); results.append(dict(width=w,robust=robust,parameters=sum(p.numel() for p in m.parameters()),validation=val)); print(results[-1],flush=True)
   if score>best: best=score; winner=m; selected=results[-1].copy()
 test=evaluate(winner,c); torch.save(winner.state_dict(),'student-v5.pt'); torch.jit.trace(winner.eval(),torch.zeros(1,1,8,8)).save('student-v5.torchscript.pt'); np.savez('split-v5.npz',train=a,validation=b,test=c)
 with open('evidence-v5.json','w') as f: json.dump(dict(candidates=results,selected=selected,test=test,split=[len(a),len(b),len(c)],epsilon=.12,attack_steps=10,seed=42),f,indent=2)
if __name__=='__main__': main()

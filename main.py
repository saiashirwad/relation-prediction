import openke
from openke.config import Trainer, Tester
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

from models import RotAtte

import torch 
import numpy as np 

negative_rate = 10 
batch_size = 1000

train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K237/", 
	batch_size = batch_size,
	threads = 1,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = negative_rate,
	neg_rel = 0
)

facts = TrainDataLoader(
	in_path = "./benchmarks/FB15K237/", 
	batch_size = train_dataloader.get_triple_tot(),
	threads = 1,
	sampling_mode = "normal", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 0,
	neg_rel = 0
)

h, t, r, _, _ = [f for f in facts][0].values()
h = torch.Tensor(h).to(torch.long)
t = torch.Tensor(t).to(torch.long)
r = torch.Tensor(r).to(torch.long)

facts = torch.stack((h, r, t)).cuda().t()
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

n_ent = train_dataloader.get_ent_tot()
n_rel = train_dataloader.get_rel_tot()

in_dim = 50
out_dim = 50

print(train_dataloader.get_batch_size())
print(batch_size)

rotatte = RotAtte(n_ent, n_rel, in_dim, out_dim, facts, 1, 1, 20, batch_size=batch_size, device="cuda")
model = NegativeSampling(model=rotatte, loss=SigmoidLoss(adv_temperature=2), batch_size=train_dataloader.get_batch_size(), regul_rate=0.0)

trainer = Trainer(model=model, data_loader=train_dataloader, opt_method="adam")
trainer.run(lr=2e-3, train_times=10)


tester = Tester(model=rotatte, data_loader=test_dataloader, use_gpu=True)
result = tester.run_link_prediction(type_constrain=False)
print(result)
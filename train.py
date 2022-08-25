from dataset import CustomImageDataLoader, CustomImageDataset
from add_couchbase import ImagesDataBase, Attempt
from models import *
import torch


class Train:
	def __init__(self, tc: TrainCommands, model: nn.Module, cidl: CustomImageDataLoader, cid: CustomImageDataset, db: ImagesDataBase):
		self.loader = cidl(tc, cid, db)
		self.model = model(tc)

		criterion_dict = tc.criterion
		criterion_name = criterion_dict.pop('name')
		if len(criterion_dict.keys()) == 0:
			self.criterion = getattr(nn, criterion_name)()
		else:
			self.criterion = getattr(nn, criterion_name)(**criterion_dict)
		
		optimizer_dict = tc.optimizer
		optimizer_name = optimizer_dict.pop('name')
		if len(optimizer_dict.keys()) == 0:
			self.optimizer = getattr(torch.optim, optimizer_name)(self.model.parameters())
		else:
			self.optimizer = getattr(torch.optim, optimizer_name)(self.model.parameters(), **optimizer_dict)
		self.n_epoch = tc.n_epoch
		self.model_name = tc.model_name

	def train(self):
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		Epoch = [x for x in range(0,self.n_epoch)]
		Loss = [0] * self.n_epoch
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
		for epoch in range(self.n_epoch):
			running_loss = 0.0
			inputs, labels = self.loader.iterate_training()
			inputs, labels = inputs.to(device), labels.to(device)
			self.optimizer.zero_grad()
			if torch.cuda.is_available():
				self.model.cuda()
				outputs = self.model(inputs).to(device)
			else:
				outputs = self.model(inputs)
			loss = self.criterion(outputs, labels.squeeze())

			loss.backward()
			self.optimizer.step()
			running_loss = running_loss + loss.item()
			scheduler.step(running_loss)
			
			from main import find_attempt, update_attempt
			a = find_attempt(name = self.model_name)
			a['training_losses'].append(running_loss)
			a = Attempt(**a)
			update_attempt(a)

			if epoch % 5 == 4:
				print(f'[Epoch: {epoch + 1}, Progress: {((epoch+1)*100/self.n_epoch):.3f}%] loss: {running_loss:.6f}')

			running_loss = 0.0
			
		torch.save(self.model.state_dict(), "models/{}.pt".format(self.model_name))
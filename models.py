from commands import TrainCommands, ValidationTestCommands
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch

class CNN(nn.Module):
	def __init__(self, tc: TrainCommands | ValidationTestCommands):
		super().__init__()
		n_labels = tc.n_labels
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.flatten = nn.Flatten(start_dim=1)
		self.fc1 = nn.Linear(16*122*122, 120) # Manually calculated I will explain next week
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, n_labels) #unique label size

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.flatten(x)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class CNNSeq(nn.Module):
	def __init__(self, tc: TrainCommands | ValidationTestCommands):
		super(CNNSeq, self).__init__()
		n_labels = tc.n_labels
		self.layers = nn.Sequential(
			# First 2D convolution layer
			nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# Second 2D convolution layer
			nn.Conv2d(in_channels= 6, out_channels = 16, kernel_size=5),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# Linear Layers
			nn.Flatten(start_dim=1),
			nn.Linear(in_features = 16*122*122, out_features = 120),
			nn.ReLU(inplace=True),
			nn.Linear(in_features = 120, out_features = 84),
			nn.ReLU(inplace=True),
			nn.Linear(in_features = 84,out_features = n_labels)
		)
	
	def forward(self, x):
		x = self.layers(x)
		return x

class CustomModel(nn.Module):
	def __init__(self, tc: ValidationTestCommands | TrainCommands):
		super(CustomModel, self).__init__()
		n_labels = tc.n_labels
		layer_commands = tc.model['layers']
		layer_list = []
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		def autosize(layer_name, index, layer_commands):
			for k in reversed(range(0,len(layer_commands))):
				layer = layer_commands[k]
				if k < index and layer['name'] == layer_name :
					value = [val for key, val in layer.items() if "out" in key][-1]
					return value

		for i in range(0, len(layer_commands)):
			layer = layer_commands[i]
			layer_name = layer['name']
			string_operation = "nn.{}(".format(layer_name)
			for j in range(1,len(layer.keys())):
				key = list(layer.keys())[j]
				if layer[key] == "n_labels":
					string_operation = string_operation + "{}={},".format(key, n_labels)
				elif layer[key] == "auto":
					value = autosize(layer_name, i, layer_commands)
					string_operation = string_operation + "{}={},".format(key, value)
				else:
					string_operation = string_operation + "{}={},".format(key, layer[key])
			string_operation = string_operation[0:-1] + ").to(device)"
			if len(layer) == 1:
				string_operation = "nn.{}().to(device)".format(layer_name)

			operation = eval(string_operation)
			layer_list.append(operation)
		
		self.layers = nn.Sequential(*layer_list).to(device)

	def forward(self, x):
		x = self.layers(x)
		return x
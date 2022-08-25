from add_couchbase import ImagesDataBase
from torch.utils.data import Dataset
from torchvision import transforms
from commands import TrainCommands, ValidationTestCommands
from PIL import Image
import numpy as np
import warnings
import random
import base64
import torch
import io

class CustomImageDataset(Dataset):
	def __init__(self, tc: TrainCommands | ValidationTestCommands , db: ImagesDataBase):
		transform = tc.transform
		target_transform = tc.target_transform
		self.image_ops = tc.image_ops
		self.label_names = tc.label_names
		tc.n_labels = len(self.label_names)
		
		self.db = db
		
		def create_transforms(transforms_dict):
			transforms_list = []
			for operation in transforms_dict:
				if type(transforms_dict[operation]) == bool:
					string_operation = "transforms.{}()".format(operation)
				elif type(transforms_dict[operation]) == dict:
					string_operation = "transforms.{}(".format(operation)
					for param in transforms_dict[operation]:
						string_operation = string_operation + "{}={},".format(param, transforms_dict[operation][param])
					string_operation = string_operation[0:-1] + ")"
				transforms_list.append(eval(string_operation))
			return transforms_list

		if transform != None and transform != {}:
			transforms_list = create_transforms(transform)
			self.transform = transforms.Compose(transforms_list)
		else:
			self.transform == False

		if target_transform != None and target_transform != {}:
			transforms_list = create_transforms(target_transform)
			self.target_transform = transforms.Compose(transforms_list)
		else:
			self.target_transform == False

	def get_item_by_id(self, key):
		while True:
			try:
				image_dict = self.db.get_image_by_key(key)
				buf = base64.b64decode(image_dict['base64'])
				buf = io.BytesIO(buf)
				img = Image.open(buf)
				
				label = image_dict['classification']
				label_arr = np.full((len(self.label_names), 1), 0, dtype=float)
				label_arr[self.label_names.index(label)]= 1.0
				break
			except:
				print("Couldn't fetch the image, Retrying with another specified image")

		if self.image_ops != None and self.image_ops != []:
			for op in self.image_ops:
				for param in op:
					if type(op[param]) == bool:
						string_operation = "img.{}()".format(param)
					elif type(op[param]) == dict:
						string_operation = "img.{}(".format(param)
						for inner_param in op[param]:
							string_operation = string_operation + "{}={},".format(inner_param, op[param][inner_param])
						string_operation = string_operation[0:-1] + ")"
					
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						img = eval(string_operation)
					
		if not self.transform	== False:
			img = self.transform(img)
		
		if not self.target_transform == False:
			label = self.target_transform(label_arr)

		return img, label

	def get_random_item(self):
		while True:
			try:
				random_label_idx = random.randint(0,len(self.label_names) - 1)
				label = self.label_names[random_label_idx]
				
				label_arr = np.full((len(self.label_names), 1), 0, dtype=float)
				label_arr[random_label_idx] = 1.0
				
				image_dict = self.db.random_lookup_by_classification(self.db(),label)["$1"]
				buf = base64.b64decode(image_dict['base64'])
				buf = io.BytesIO(buf)
				img = Image.open(buf)
				break
			except:
				print("Couldn't fetch the image, Retrying with another random image")

		if self.image_ops != None and self.image_ops != []:
			for op in self.image_ops:
				for param in op:
					if type(op[param]) == bool:
						string_operation = "img.{}()".format(param)
					elif type(op[param]) == dict:
						string_operation = "img.{}(".format(param)
						for inner_param in op[param]:
							string_operation = string_operation + "{}={},".format(inner_param, op[param][inner_param])
						string_operation = string_operation[0:-1] + ")"
					
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						img = eval(string_operation)
					
		if not self.transform	== False:
			img = self.transform(img)
		
		if not self.target_transform == False:
			label = self.target_transform(label_arr)
		
		return img, label

class CustomImageDataLoader:
	def __init__(self, tc: TrainCommands, cid: CustomImageDataset, db: ImagesDataBase):
		self.batch_size = tc.batch_size
		self.cid = cid(tc, db)
	
	def iterate_training(self):
		train_features = []
		train_labels = []
		for i in range(0,self.batch_size):
			img, label = self.cid.get_random_item()
			train_features.append(img)
			train_labels.append(label)
		train_features = [t.numpy() for t in train_features]
		train_features = np.asarray(train_features, dtype='float64')
		train_features = torch.from_numpy(train_features).float()

		train_labels = [t.numpy() for t in train_labels]
		train_labels = np.asarray(train_labels, dtype='float64')
		train_labels = torch.from_numpy(train_labels).float()

		return train_features, train_labels
	


	




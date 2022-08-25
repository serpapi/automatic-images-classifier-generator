from pydantic import BaseModel

class TrainCommands(BaseModel):
	model_name: str = "american_dog_species"
	criterion: dict = {"name": "CrossEntropyLoss"}
	optimizer: dict = {"name": "SGD", "lr": 0.001, "momentum": 0.9 }
	batch_size: int = 4
	n_epoch: int = 100
	n_labels: int = None
	image_ops: list = [{"resize":{"size": (500, 500), "resample": "Image.ANTIALIAS"}}, {"convert": {"mode": "'RGB'"}}]
	transform: dict = {"ToTensor": True, "Normalize": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}}
	target_transform: dict = {"ToTensor": True}
	label_names: list = [
    "American Hairless Terrier imagesize:500x500",
    "Alaskan Malamute imagesize:500x500",
    "American Eskimo Dog imagesize:500x500",
    "Australian Shepherd imagesize:500x500",
    "Boston Terrier imagesize:500x500",
    "Boykin Spaniel imagesize:500x500",
    "Chesapeake Bay Retriever imagesize:500x500",
    "Catahoula Leopard Dog imagesize:500x500",
    "Toy Fox Terrier imagesize:500x500"
  ]
	model: dict = {
		"name": "",
		"layers": [
			{
				"name": "Conv2d",
				"in_channels": 3,
				"out_channels": 6,
				"kernel_size": 5
			},
			{
				"name": "ReLU",
				"inplace": True,
			},
			{
				"name": "MaxPool2d",
				"kernel_size": 2,
				"stride": 2
			},
			{
				"name": "Conv2d",
				"in_channels": "auto",
				"out_channels": 16,
				"kernel_size": 5
			},
			{
				"name": "ReLU",
				"inplace": True,
			},
			{
				"name": "MaxPool2d",
				"kernel_size": 2,
				"stride": 2
			},
			{
				"name": "Conv2d",
				"in_channels": "auto",
				"out_channels": 32,
				"kernel_size": 5
			},
			{
				"name": "ReLU",
				"inplace": True,
			},
			{
				"name": "MaxPool2d",
				"kernel_size": 2,
				"stride": 2
			},
			{
				"name": "Flatten",
				"start_dim": 1
			},
			{
				"name": "Linear",
				"in_features": 32*59*59,
				"out_features": 120
			},
			{
				"name": "ReLU",
				"inplace": True,
			},
			{
				"name": "Linear",
				"in_features": "auto",
				"out_features": 84
			},
			{
				"name": "ReLU",
				"inplace": True,
			},
			{
				"name": "Linear",
				"in_features": "auto",
				"out_features": "n_labels"
			}
		]
	}

class ValidationTestCommands(BaseModel):
	ids: list | None = None
	limit: int = 200
	label_names: list = [
		"American Hairless Terrier imagesize:500x500",
    "Alaskan Malamute imagesize:500x500",
    "American Eskimo Dog imagesize:500x500",
    "Australian Shepherd imagesize:500x500",
    "Boston Terrier imagesize:500x500",
    "Boykin Spaniel imagesize:500x500",
    "Chesapeake Bay Retriever imagesize:500x500",
    "Catahoula Leopard Dog imagesize:500x500",
    "Toy Fox Terrier imagesize:500x500"
	]
	n_labels: int = None
	criterion: dict = {"name": "CrossEntropyLoss"}
	model_name: str = "american_dog_species"
	image_ops: list = [{"resize":{"size": (500, 500), "resample": "Image.ANTIALIAS"}}, {"convert": {"mode": "'RGB'"}}]
	transform: dict = {"ToTensor": True, "Normalize": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}}
	target_transform: dict = {"ToTensor": True}
	model: dict = {
		"name": "",
		"layers": [
			{
				"name": "Conv2d",
				"in_channels": 3,
				"out_channels": 6,
				"kernel_size": 5
			},
			{
				"name": "ReLU",
				"inplace": True,
			},
			{
				"name": "MaxPool2d",
				"kernel_size": 2,
				"stride": 2
			},
			{
				"name": "Conv2d",
				"in_channels": "auto",
				"out_channels": 16,
				"kernel_size": 5
			},
			{
				"name": "ReLU",
				"inplace": True,
			},
			{
				"name": "MaxPool2d",
				"kernel_size": 2,
				"stride": 2
			},
			{
				"name": "Conv2d",
				"in_channels": "auto",
				"out_channels": 32,
				"kernel_size": 5
			},
			{
				"name": "ReLU",
				"inplace": True,
			},
			{
				"name": "MaxPool2d",
				"kernel_size": 2,
				"stride": 2
			},
			{
				"name": "Flatten",
				"start_dim": 1
			},
			{
				"name": "Linear",
				"in_features": 32*59*59,
				"out_features": 120
			},
			{
				"name": "ReLU",
				"inplace": True,
			},
			{
				"name": "Linear",
				"in_features": "auto",
				"out_features": 84
			},
			{
				"name": "ReLU",
				"inplace": True,
			},
			{
				"name": "Linear",
				"in_features": "auto",
				"out_features": "n_labels"
			}
		]
	}
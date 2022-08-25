from tkinter import W
from fastapi.testclient import TestClient
from main import app
from add_couchbase import *
from models import *
from commands import *
from torchvision import transforms
from PIL import Image
import warnings
import json
import io
import base64

client = TestClient(app)

def test_read_root():
	response = client.get("/")
	assert response.status_code == 200
	assert response.json() == {"Hello": "World"}

def test_add_to_db_behaviour():
	# Get The first link
	query = Query(q="Coffee")
	db = ImagesDataBase()
	serpapi = Download(query, db)
	serpapi.serpapi_search()
	link = serpapi.results[0]

	# Call the endpoint to check if an object with a specific link has been added to database and delete it
	body = json.dumps({"q":"Coffee", "limit": 1})
	response = client.post("/add_to_db", headers = {"Content-Type": "application/json"}, data=body, allow_redirects=True)
	assert response.status_code == 200
	assert db.check_if_it_exists(link, "Coffee") != None
	db.delete_by_link(link)

def test_multiple_query_behaviour():
	# Get The first link
	chips = "q:coffee,g_1:espresso:AKvUqs_4Mro%3D"
	query = Query(q="Coffee imagesize:500x500", chips=chips)
	db = ImagesDataBase()
	serpapi = Download(query, db)
	serpapi.serpapi_search()
	link = serpapi.results[0]

	# Call the endpoint to check if an object with a specific link has been added to database and delete it
	multiple_queries = MultipleQueries(queries=["Coffee"], desired_chips_name="espresso", limit=1)
	body = json.dumps(multiple_queries.dict())
	response = client.post("/multiple_query", headers = {"Content-Type": "application/json"}, data=body, allow_redirects=True)
	assert response.status_code == 200
	assert db.check_if_it_exists(link, "Coffee imagesize:500x500") != None
	db.delete_by_link(link)

def test_optimizers():
	optimizers = [
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			optimizer = {
				"name": "Adadelta",
				"lr": 1.0,
				"rho": 0.9,
				"eps": 1e-06,
				"weight_decay": 0.1
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			optimizer = {
				"name": "Adagrad",
				"lr": 1.0,
				"lr_decay": 0.1,
				"weight_decay": 0.1,
				"initial_accumulator_value": 0.1,
				"eps": 1e-10
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			optimizer = {
				"name": "Adam",
				"lr": 1.0,
				"betas": (0.9, 0.999),
				"eps": 1e-10,
				"weight_decay": 0.1,
				"amsgrad": True
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			optimizer = {
				"name": "AdamW",
				"lr": 1.0,
				"betas": (0.9, 0.999),
				"eps": 1e-10,
				"weight_decay": 0.1,
				"amsgrad": True
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			optimizer = {
				"name": "Adamax",
				"lr": 1.0,
				"betas": (0.9, 0.999),
				"eps": 1e-10,
				"weight_decay": 0.1
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			optimizer = {
				"name": "ASGD",
				"lr": 1.0,
				"lambd": 0.1,
				"alpha": 0.82,
				"t0": 1000000.0,
				"weight_decay": 0.1
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			optimizer = {
				"name": "NAdam",
				"lr": 1.0,
				"betas": (0.9, 0.999),
				"eps": 1e-09,
				"weight_decay": 0.1,
				"momentum_decay": 0.004
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			optimizer = {
				"name": "RAdam",
				"lr": 1.0,
				"betas": (0.9, 0.999),
				"eps": 1e-09,
				"weight_decay": 0.1,
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			optimizer = {
				"name": "RMSprop",
				"lr": 1.0,
				"alpha": 0.99,
				"eps": 1e-09,
				"weight_decay": 0.1,
				"momentum": 0.1,
				"centered": True
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			optimizer = {
				"name": "Rprop",
				"lr": 1.0,
				"etas": (0.5, 1.2),
				"step_sizes": (1e-6, 50)
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			optimizer = {
				"name": "SGD",
				"lr": 1.0,
				"momentum": 0.1,
				"dampening": 0,
				"weight_decay": 0.1,
				"nesterov": True
			}
		)
	]

	for optimizer in optimizers:
		# Train a Model for 1 Epoch
		body = json.dumps(optimizer.dict())
		response = client.post("/train", headers = {"Content-Type": "application/json"}, data=body, allow_redirects = True)
		assert response.status_code == 200, "{}".format(optimizer.optimizer['name'])

		# Find The Training Attempt in the Database
		find_response = client.post("/find_attempt?name=unit_test_model", allow_redirects = True)
		assert find_response.status_code == 200
		assert type(find_response) != None

		# Delete The Training Attempt in the Database to make the Succeeding Unit Test Valid
		delete_response = client.post("/delete_attempt?name=unit_test_model", allow_redirects = True)
		assert delete_response.status_code == 200

def test_criterions():
	criterions = [
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			criterion = {
				"name": "L1Loss",
				"reduction": "sum"
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			criterion = {
				"name": "MSELoss",
				"reduction": "sum"
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			criterion = {
				"name": "CrossEntropyLoss",
				"reduction": "sum",
				"label_smoothing": 0.1
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			criterion = {
				"name": "PoissonNLLLoss",
				"log_input": True,
				"full": True,
				"eps": 1e-9,
				"reduction": "sum"
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			criterion = {
				"name": "KLDivLoss",
				"reduction": "sum",
				"log_target": True
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			criterion = {
				"name": "BCEWithLogitsLoss",
				"reduction": "sum"
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			criterion = {
				"name": "HingeEmbeddingLoss",
				"margin": 1.0,
				"reduction": "sum"
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			criterion = {
				"name": "HuberLoss",
				"reduction": "sum",
				"delta": 1.1
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			criterion = {
				"name": "SmoothL1Loss",
				"reduction": "sum",
				"beta": 1.1
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			criterion = {
				"name": "SoftMarginLoss",
				"reduction": "sum"
			}
		),
		TrainCommands(
			n_epoch = 1,
			batch_size = 2,
			model_name = "unit_test_model",
			criterion = {
				"name": "MultiLabelSoftMarginLoss",
				"reduction": "sum"
			}
		)
	]

	for criterion in criterions:
		# Train a Model for 1 Epoch
		body = json.dumps(criterion.dict())
		response = client.post("/train", headers = {"Content-Type": "application/json"}, data=body, allow_redirects = True)
		assert response.status_code == 200, "{}".format(criterion.criterion['name'])

		# Find The Training Attempt in the Database
		find_response = client.post("/find_attempt?name=unit_test_model", allow_redirects = True)
		assert find_response.status_code == 200
		assert type(find_response) != None

		# Delete The Training Attempt in the Database to make the Succeeding Unit Test Valid
		delete_response = client.post("/delete_attempt?name=unit_test_model", allow_redirects = True)
		assert delete_response.status_code == 200

def test_convolutional_layers():
	layers = [
		TrainCommands(model={"layers":[
			{
				"name":"Conv1d",
				"in_channels":16,
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'reflect'",
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"Conv2d",
				"in_channels":16,
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'reflect'",
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"Conv3d",
				"in_channels":16,
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'reflect'",
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"ConvTranspose1d",
				"in_channels":16,
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'zeros'"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"ConvTranspose2d",
				"in_channels":16,
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'zeros'"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"ConvTranspose3d",
				"in_channels":16,
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'zeros'"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"LazyConv1d",
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'zeros'"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"LazyConv2d",
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'zeros'"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"LazyConv3d",
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'zeros'"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"LazyConvTranspose1d",
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'zeros'"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"LazyConvTranspose2d",
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'zeros'"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"LazyConvTranspose3d",
				"out_channels":33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1,
				"groups": 1,
				"bias": True,
				"padding_mode": "'zeros'"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"Unfold",
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name":"Fold",
				"output_size": 33,
				"kernel_size":3,
				"stride":2,
				"padding":1,
				"dilation":1
			}
		]}),
	]

	for layer in layers:
		assert type(CustomModel(tc=layer)) == CustomModel

def test_pooling_layers():
	layers = [
		TrainCommands(model={"layers":[
			{
				"name": "MaxPool1d",
				"kernel_size": 3,
				"stride": 2,
				"padding": 1,
				"dilation": 1,
				"return_indices": True,
				"ceil_mode": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "MaxPool2d",
				"kernel_size": 3,
				"stride": 2,
				"padding": 1,
				"dilation": 1,
				"return_indices": True,
				"ceil_mode": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "MaxPool3d",
				"kernel_size": 3,
				"stride": 2,
				"padding": 1,
				"dilation": 1,
				"return_indices": True,
				"ceil_mode": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "MaxUnpool1d",
				"kernel_size": 3,
				"stride": 2,
				"padding": 1
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "MaxUnpool2d",
				"kernel_size": 3,
				"stride": 2,
				"padding": 1
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "MaxUnpool3d",
				"kernel_size": 3,
				"stride": 2,
				"padding": 1
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "AvgPool1d",
				"kernel_size": 3,
				"stride": 2,
				"padding": 1,
				"ceil_mode": True,
				"count_include_pad": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "AvgPool2d",
				"kernel_size": 3,
				"stride": 2,
				"padding": 1,
				"ceil_mode": True,
				"count_include_pad": True,
				"divisor_override": 1
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "AvgPool3d",
				"kernel_size": 3,
				"stride": 2,
				"padding": 1,
				"ceil_mode": True,
				"count_include_pad": True,
				"divisor_override": 1
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "FractionalMaxPool2d",
				"kernel_size": 3,
				"output_ratio": 0.1,
				"return_indices": True 
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "FractionalMaxPool3d",
				"kernel_size": 3,
				"output_ratio": 0.1,
				"return_indices": True 
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "FractionalMaxPool3d",
				"kernel_size": 3,
				"output_size": 4,
				"return_indices": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "AdaptiveMaxPool1d",
				"output_size": 3,
				"return_indices": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "AdaptiveMaxPool2d",
				"output_size": 3,
				"return_indices": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "AdaptiveMaxPool3d",
				"output_size": 3,
				"return_indices": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "AdaptiveAvgPool1d",
				"output_size": 3
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "AdaptiveAvgPool2d",
				"output_size": 3
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "AdaptiveAvgPool3d",
				"output_size": 3
			}
		]})
	]

	for layer in layers:
		assert type(CustomModel(tc=layer)) == CustomModel

def test_linear():
	layers = [
		TrainCommands(model={"layers":[
			{
				"name": "Linear",
				"in_features": 5,
				"out_features": 6,
				"bias": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Bilinear",
				"in1_features": 5,
				"in2_features": 6,
				"out_features": 7,
				"bias": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "LazyLinear",
				"out_features": 6,
				"bias": True
			}
		]}),
	]

	for layer in layers:
		assert type(CustomModel(tc=layer)) == CustomModel

def test_utilities():
	layers = [
		TrainCommands(model={"layers":[
			{
				"name": "Flatten",
				"start_dim": 1,
				"end_dim": -1,
			}
		]})
	]

	for layer in layers:
		assert type(CustomModel(tc=layer)) == CustomModel

def test_non_linear_activations():
	layers = [
		TrainCommands(model={"layers":[
			{
				"name": "ELU",
				"alpha": 1.1,
				"inplace": True,
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Hardshrink",
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Hardsigmoid",
				"inplace": True,
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Hardtanh",
				"min_val": -2.0,
				"max_val": 2.0,
				"inplace": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Hardswish",
				"inplace": True,
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "LeakyReLU",
				"negative_slope": 0.001,
				"inplace": True,
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "LogSigmoid"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "MultiheadAttention",
				"embed_dim": 8,
				"num_heads": 4,
				"dropout": 0.1,
				"bias": True,
				"add_bias_kv": True,
				"add_zero_attn": True,
				"kdim": 6,
				"vdim": 7,
				"batch_first": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "PReLU",
				"num_parameters": 2,
				"init": 0.5
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "ReLU",
				"inplace": True,
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "ReLU6",
				"inplace": True,
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "RReLU",
				"lower": 0.25,
				"upper": 0.30,
				"inplace": True,
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "SELU",
				"inplace": True,
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "CELU",
				"alpha": 0.001,
				"inplace": True,
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "GELU"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Sigmoid",
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "SiLU",
				"inplace": True,
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Mish",
				"inplace": True,
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Softplus",
				"beta": 0.1,
				"threshold": 25
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Softshrink"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Softsign"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Tanh"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Tanhshrink"
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "Threshold",
				"threshold": 0.1,
				"value": 25,
				"inplace": True
			}
		]}),
		TrainCommands(model={"layers":[
			{
				"name": "GLU",
				"dim": -2
			}
		]}),
	]

	for layer in layers:
		assert type(CustomModel(tc=layer)) == CustomModel

def test_transforms():
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
	
	transforms_dicts = [
		{
			"CenterCrop": {
				"size": (5,5)
			}
		},
		{
			"ColorJitter": {
				"brightness": (0.0, 0.5),
				"contrast": (0.0,0.5),
				"saturation": (0.0,0.5),
				"hue": (0.0,0.5)
			}
		},
		{
			"FiveCrop": {
				"size": (5,5)
			}
		},
		{
			"Grayscale": {
				"num_output_channels": 5
			}
		},
		{
			"Pad": {
				"padding": 5,
				"fill": 0,
				"padding_mode": "'constant'"
			}
		},
		{
			"RandomAffine": {
				"degrees": 5,
				"translate": (0 ,0.8),
				"scale": (50,50),
				"shear": 5,
				"interpolation": "'InterpolationMode.BILINEAR'",
				"fill": 0.1
			}
		},
		{
			"RandomCrop": {
				"size": 1,
				"padding": 5,
				"pad_if_needed": True,
				"fill": 0,
				"padding_mode": "'constant'"
			}
		},		
		{
			"RandomGrayscale": {
				"p": 0.2
			}
		},
		{
			"RandomHorizontalFlip": {
				"p": 0.2
			}
		},
		{
			"RandomPerspective": {
				"distortion_scale": 0.2,
				"p": 0.5,
				"interpolation": "'InterpolationMode.BILINEAR'",
				"fill": 0
			}
		},
		{
			"RandomResizedCrop": {
				"size": (5,5),
				"scale": (0.07, 0.9),
				"ratio": (0.76, 1.34),
				"interpolation": "'InterpolationMode.BILINEAR'"
			}
		},
		{
			"RandomRotation": {
				"degrees": 5,
				"interpolation": "'InterpolationMode.BILINEAR'",
				"expand": False,
				"center": (5,5),
				"fill": 0
			}
		},
		{
			"RandomVerticalFlip": {
				"p": 0.4
			}
		},
		{
			"Resize": {
				"size": (5,5),
				"interpolation": "'InterpolationMode.BILINEAR'",
				"max_size": 5,
				"antialias": True
			}
		},
		{
			"TenCrop": {
				"size": (5,5),
				"vertical_flip": True
			}
		},
		{
			"GaussianBlur": {
				"kernel_size": 5,
				"sigma": (0.1, 2.0)
			}
		},
		{
			"RandomInvert": {
				"p": 0.5
			}
		},
		{
			"RandomPosterize": {
				"bits": 5,
				"p": 0.5
			}
		},
		{
			"RandomSolarize": {
				"threshold": 0.1,
				"p": 0.5
			}
		},
		{
			"RandomAdjustSharpness": {
				"sharpness_factor": 0.1,
				"p": 0.5
			}
		},
		{
			"RandomAutocontrast": {
				"p": 0.5
			}
		},
		{
			"RandomEqualize": {
				"p": 0.5
			}
		},
		{
			"Normalize": {
				"mean": (0.5, 0.5, 0.5),
				"std": (0.5, 0.5, 0.5),
				"inplace": True
			}
		},
		{
			"RandomErasing": {
				"p": 0.5,
				"scale": (0.2, 0.3),
				"ratio": (0.3, 3.3),
				"value": (0, 0, 0),
				"inplace": True
			}
		},
		{
			"ToPILImage": {
				"mode": "'RGB'"
			}
		},
		{
			"ToTensor": True
		},
		{
			"PILToTensor": True
		},
		{
			"RandAugment": {
				"num_ops": 3,
				"magnitude": 10,
				"num_magnitude_bins": 32,
				"interpolation": "'InterpolationMode.NEAREST'",
				"fill": [0.1, 0.2]
			}
		},
		{
			"TrivialAugmentWide": {
				"num_magnitude_bins": 32,
				"interpolation": "'InterpolationMode.NEAREST'",
				"fill": [0.1, 0.2]
			}
		}
	]

	for transforms_dict in transforms_dicts:
		transforms_list = create_transforms(transforms_dict)
		assert len(transforms_list) > 0
		assert type(transforms.Compose(transforms_list)) == transforms.Compose

def test_image():
	example = "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"
	example = base64.b64decode(example)
	example = io.BytesIO(example)
	img = Image.open(example)

	def create_image_ops(image_ops_dict, img):
		image_ops_list = []
		for operation in image_ops_dict:
			if type(image_ops_dict[operation]) == bool:
				string_operation = "img.{}()".format(operation)
			elif type(image_ops_dict[operation]) == dict:
				string_operation = "img.{}(".format(operation)
				for param in image_ops_dict[operation]:
					string_operation = string_operation + "{}={},".format(param, image_ops_dict[operation][param])
				string_operation = string_operation[0:-1] + ")"
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				img = eval(string_operation)
		return img
	
	image_ops_dicts = [
		{
			"convert": {
				"mode": "'L'",
				"palette": "'Palette.WEB'",
				"colors": 256
			}
		},
		{
			"convert": {
				"matrix": "'RGB'",
			}
		},
		{
			"convert": {
				"mode": "'L'",
			}
		},
		{
			"convert": {
				"dither": "'Dither.FLOYDSTEINBERG'"
			}
		},
		{
			"crop": {
				"box": (20, 20, 100, 100)
			}
		},
		{
			"effect_spread": {
				"distance": 5
			}
		},
		{
			"getchannel": {
				"channel": 0
			}
		},
		{
			"reduce": {
				"factor": 1,
				"box": (0,0,50,50)
			}
		},
    {
      "resize": {
        "size": (500,500),
        "resample": "Image.ANTIALIAS",
				"box": (0,0,50,50),
				"reducing_gap": 1.1
      }
    },
		{
			"rotate": {
				"angle": 5,
				"resample": 0,
				"expand": 1,
				"center": (5,5),
				"translate": (10,10),
				"fillcolor": 125
			}
		},
		{
			"transpose": {
				"method": 0
			}
		}
	]

	for image_ops_dict in image_ops_dicts:
		operated_image = create_image_ops(image_ops_dict, img)
		assert type(operated_image) == Image.Image, "{}".format(list(image_ops_dict.keys())[0])

def test_validationtests():
	train_commands = TrainCommands(batch_size = 2, n_epoch = 1, model_name = "unit_test_model")

	# Train a Model for 1 Epoch
	body = json.dumps(train_commands.dict())
	response = client.post("/train", headers = {"Content-Type": "application/json"}, data=body, allow_redirects = True)
	assert response.status_code == 200

	# Find The Training Attempt in the Database
	find_response = client.post("/find_attempt?name=unit_test_model", allow_redirects = True)
	assert find_response.status_code == 200
	assert type(find_response) != None

	# Test the Trained Model with one Image
	test_commands = ValidationTestCommands(limit = 1)
	body = json.dumps(train_commands.dict())
	response = client.post("/test", headers = {"Content-Type": "application/json"}, data=body, allow_redirects = True)
	assert find_response.status_code == 200
	
	# Find The Training Attempt in the Database
	find_response = client.post("/find_attempt?name=unit_test_model", allow_redirects = True)
	assert find_response.status_code == 200
	assert find_response.json()['testing_commands'] != {}
	assert find_response.json()['status'] == "Complete"

	# Delete The Training Attempt in the Database to make the Succeeding Unit Test Valid
	delete_response = client.post("/delete_attempt?name=unit_test_model", allow_redirects = True)
	assert delete_response.status_code == 200
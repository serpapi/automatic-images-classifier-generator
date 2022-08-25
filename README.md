# Generate machine learning models fully automatically to classify any images using SERP data

`automatic-images-classifier-generator` is a machine learning tool written in Python using [SerpApi](https://serpapi.com), Pytorch, FastAPI, and Couchbase to provide automated large dataset creation, automated training and testing of deep learning models with the ability to tweak algorithms, storing the structure and results of neural networks all in one place.

Disclaimer: This open-source machine learning software is not one of [the product offerings provided by SerpApi](https://serpapi.com/libraries). The software is using one of the product offerings, [SerpApi’s Google Images Scraper API](https://https://serpapi.com/images-results) to automatically create datasets. You may [register to SerpApi to claim free credits](https://serpapi.com/users/sign_up). You may also see the pricing page of SerpApi to get detailed information.

- [Machine Learning Tools and Features provided by `automatic-images-classifier-generator`](#machine-learning-tools-and-features-provided-by--automatic-images-classifier-generator-)
- [Installation](#installation)
- [Basic Usage of Machine Learning Tools](#basic-usage-of-machine-learning-tools)
- [Adding SERP Images to Storage Server](#adding-serp-images-to-storage-server)
  * [add_to_db](#add_to_db)
  * [multiple_query](#multiple_query)
    + [Example Dictionary](#example-dictionary)
- [Training a Model](#training-a-model)
  * [train](#train)
    + [Example Dictionary](#example-dictionary-1)
- [Testing a Model](#testing-a-model)
  * [test](#test)
    + [Example Dictionary](#example-dictionary-2)
- [Getting Information on the Training and Testing of the Model](#getting-information-on-the-training-and-testing-of-the-model)
  * [find_attempt](#find_attempt)
    + [Example Output Dictionary](#example-output-dictionary)
- [Support for Various Elements](#support-for-various-elements)
  * [Layers](#layers)
  * [Optimizers](#optimizers)
  * [Loss Functions](#loss-functions)
  * [Transforms](#transforms)
  * [Image Operations](#image-operations)
- [Keypoints for the State of the Machine Learning Tool and Its Future Roadmap](#keypoints-for-the-state-of-the-machine-learning-tool-and-its-future-roadmap)

---

# Machine Learning Tools and Features provided by `automatic-images-classifier-generator`

- Machine Learning Tools for automatic large image datasets creation powered by [SerpApi’s Google Images Scraper API](https://serpapi.com/users/sign_up)

- Machine Learning Tools for automatically training deep learning models with customized tweaks for various algorithms

- Machine Learning Tools for automatically testing machine learning models

- Machine Learning Tools for customizing nodes within pipelines of ml models, changing dimensionality of machine learning algorithms, etc.

- Machine Learning Tools for keeping the record of the training losses, employed datasets, structures of neural networks, and accuracy reports

- Async Training and Testing of Machine Learning Models

- Delivery of data necessary to create a visualization for cross-comparing different machine learning models with subtle changes in their neural network structure.

- Various shortcuts for preprocessing with targeted data mining of SERP data

---

# Installation

1) Clone the repository
```
gh repo clone serpapi/automatic-images-classifier-generator
```

2) [Open a SerpApi Account (Free Credits Available upon Registration)](https://serpapi.com/users/sign_up)

3) [Download and Install Couchbase](https://www.couchbase.com/downloads)

4) Head to Server Dashboard URL (Ex: `http://kagermanov:8091`), and create a bucket named `images`

![image](https://user-images.githubusercontent.com/73674035/186512765-048da222-c86b-4304-8456-5ae9bd6a8c8a.png)

5) Install required Python Libraries
```
pip install -r requirements.txt
```

6) Fill `credentials.py` file with your server credentials, and [SerpApi credentials](https://serpapi.com/manage-api-key)

7) Run Setup Server File
```
python setup_server.py
```

8) Run the FastAPI Server
```
uvicorn main:app --host 0.0.0.0 --port 8000
```
or you may simply use a debugging server by clicking on `main.py` and running a degugging server:

![debug](https://user-images.githubusercontent.com/73674035/186514308-c8760bcd-3467-4255-893f-b327e357fb03.png)

9) Optionally run the tests:
```
pytest test_main.py
```

---

# Basic Usage of Machine Learning Tools
1) Head to `localhost:8000/docs`
2) Use `add_to_db` endpoint to call to update image database
3) Use `train` endpoint to train a model. The trained model will be saved on `models` folder when the training is complete. The training is an async process. Keep an eye out for terminal outputs for the progression.
4) Use `test` endpoint to test a model.
5) Use `find_attempt` endpoint to fetch the data on the training and testing process (losses at each epoch, accuracy etc.)

---

# Adding SERP Images to Storage Server
## `add_to_db`

User can make singular searches with [SerpApi Images Scraper API](https://serpapi.com/images-results), and automatically add them to local image storage server.

<details>
<summary>Visual Documentation Playground</summary>

Head to `http://localhost:8000/docs#/default/create_query_add_to_db__post` and customize the dictionary:
![add_to_db](https://user-images.githubusercontent.com/73674035/186532744-2b1258ca-97c7-40d4-aaeb-11cf4e6a7510.png)

</details>

<details>
<summary>Curl Command with Explanation of Parameters</summary>

```
curl -X 'POST' \
  'http://localhost:8000/add_to_db/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "google_domain": "<SerpApi Parameter: Google Domain to be scraped>",
  "limit": <External Parameter: Integer, Limit of Images to be downloaded>,
  "ijn": "<SerpApi Parameter: Page Number>",
  "q": "<SerpApi Parameter: Query to be searched for images>",
  "chips": "<SerpApi Parameter: chips parameter that specifies the image search>",
  "desired_chips_name": "<External Parameter: Specification Name for chips parameter>",
  "api_key": "<SerpApi Parameter: API Key>",
  "no_cache": <SerpApi Parameter: Choice for Cached or Live Results>
}'
```
</details>

###Example Dictionary

```py
{
  "google_domain": "google.com",
  "limit": 100,
  "ijn": 0,
  "chips": "",
  "desired_chips_name": "phone",
  "api_key": "<api_key>",
  "no_cache": True
}
```

## `multiple_query`

User can make multiple searches with SerpApi Images Scraper API, and automatically add them to local image storage server.

<details>
<summary>Visual Documentation Playground</summary>

Head to `http://localhost:8000/docs#/default/create_multiple_query_multiple_query__post` and customize the dictionary:
![multiple_query](https://user-images.githubusercontent.com/73674035/186536965-353a759b-6660-46ee-b327-ccd84775017f.png)

</details>

<details>
<summary>Curl Command with Explanation of Parameters</summary>

```
curl -X 'POST' \
  'http://localhost:8000/multiple_query/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "queries": [
    "<SerpApi Parameter: Query to be searched for images>"
    "<SerpApi Parameter: Query to be searched for images>"
    ...
  ],
  "desired_chips_name": "<External Parameter: Specification Name for chips parameter>",
  "height": <External Parameter: Integer, Height of desired images>,
  "width": <External Parameter: Integer, Width of desired images>,
  "number_of_pages": <External Parameter: Total Number of pages to be scraped for each query>,
  "google_domain": "<SerpApi Parameter: Google Domain to be scraped>",
  "api_key": "<SerpApi Parameter: API Key>",
  "limit": <External Parameter: Integer, Limit of Images to be downloaded per each query on each page>,
  "no_cache": <SerpApi Parameter: Choice for Cached or Live Results>
}'
```

</details>

### Example Dictionary
```py
{
  "queries": [
    "american foxhound",
    "german shephard",
    "caucasian shepherd"
  ],
  "desired_chips_name": "dog",
  "height": 500,
  "width": 500,
  "number_of_pages": 2,
  "google_domain": "google.com",
  "limit": 100,
  "api_key": "<api_key>",
  "no_cache": True
}
```

# Training a Model
User can train a model with a customized dictionary from `train` endpoint.

## `train`

<details>
<summary>Visual Documentation Playground</summary>

Head to `http://localhost:8000/docs#/default/train_train__post` and customize the dictionary:
![train](https://user-images.githubusercontent.com/73674035/186538215-01a15163-8775-4cdc-a760-0257f7b89507.png)

</details>

<details>
<summary>Curl Command with Explanation of Parameters</summary>

```
curl -X 'POST' \
  'http://localhost:8000/train/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_name": "< Name the user want to the model, will be saved in models/ folder with the same name>",
  "criterion": {
    "name": "<Loss Function>"
    "<Parameter of the Loss Function>": "<Value for the Parameter of a Loss Function>"
    ...
  },
  "optimizer": {
    "name": "<Optimizer Function>"
    "<Parameter of the Optimizer>": "<Value for the Parameter of an Optimizer>"
    ...
  },
  "batch_size": <How many images will be fetched at each epoch of a training>,
  "n_epoch": <Number of epochs>,
  "n_labels": 0, ### Keep it like that, it will be automatically updated in automatic training process
  "image_ops": [
    {
      "<Name of the function in PIL Image Class>": {
        "<Parameter of the function in PIL Image Class>": <Value for Parameter of the function in PIL Image Class>,
        ...
      }
    },
    ...
  ],
  "transform": {
    "<Pytorch Transforms Layer Name without parameters>": true,
    "<Pytorch Transforms Layer Name with parameters>": {
      "<Pytorch Transforms Layer Parameter>": <Value for Pytorch Transforms Layer Parameter>
      ...
    }
  },
  "target_transform": {
    "<Pytorch Transforms Layer Name without parameters for target of the operation(e.g. classification)>": true
    "<Pytorch Transforms Layer Name with parameters for target of the operation(e.g. classification)>": {
      "<Pytorch Transforms Layer Parameter for target of the operation(e.g. classification)>": <Value for Pytorch Transforms Layer Parameter for target of the operation(e.g. classification)>
      ...
    }
  },
  "label_names": [
    "<Label Name used in classification, same with the query used in adding it to database>"
    ...
  ],
  "model": {
    "name": "<Class Name of the preset model in models.py>",
    "layers": [
      {
        "name": "<Pytorch Training Layer>",
        "<Parameter in Pytorch Training Layer>": <Parameter in Pytorch Training Layer>
        ...
      },
      ...
    ]
  }
}'
```

</details>

### Example Dictionary
```py
{
  "model_name": "american_dog_species",
  "criterion": {
    "name": "CrossEntropyLoss"
  },
  "optimizer": {
    "name": "SGD",
    "lr": 0.001,
    "momentum": 0.9
  },
  "batch_size": 4,
  "n_epoch": 100,
  "n_labels": 0,
  "image_ops": [
    {
      "resize": {
        "size": [
          500,
          500
        ],
        "resample": "Image.ANTIALIAS"
      }
    },
    {
      "convert": {
        "mode": "'RGB'"
      }
    }
  ],
  "transform": {
    "ToTensor": True,
    "Normalize": {
      "mean": [
        0.5,
        0.5,
        0.5
      ],
      "std": [
        0.5,
        0.5,
        0.5
      ]
    }
  },
  "target_transform": {
    "ToTensor": True
  },
  "label_names": [
    "American Hairless Terrier imagesize:500x500",
    "Alaskan Malamute imagesize:500x500",
    "American Eskimo Dog imagesize:500x500",
    "Australian Shepherd imagesize:500x500",
    "Boston Terrier imagesize:500x500",
    "Boykin Spaniel imagesize:500x500",
    "Chesapeake Bay Retriever imagesize:500x500",
    "Catahoula Leopard Dog imagesize:500x500",
    "Toy Fox Terrier imagesize:500x500"
  ],
  "model": {
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
        "inplace": True
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
        "inplace": True
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
        "inplace": True
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
        "in_features": 111392,
        "out_features": 120
      },
      {
        "name": "ReLU",
        "inplace": True
      },
      {
        "name": "Linear",
        "in_features": "auto",
        "out_features": 84
      },
      {
        "name": "ReLU",
        "inplace": True
      },
      {
        "name": "Linear",
        "in_features": "auto",
        "out_features": "n_labels"
      }
    ]
  }
}
```
*Tips for Criterion*
- `criterion` key is responsible for calling a loss function.
- If user only provides the name of the criterion(loss function), it will be used without parameters.
- Some string inputs(especially if the user calls an external class from Pytorch), should be double quoted like `"'Parameter Value'"`.
- User may find the information on the support for [Loss Functions](#loss-functions) later in the documentation.

*Tips for Optimizer*
- `optimizer` key is responsible for calling an optimizer.
- If user only provides the name of the optimizer, it will be used without parameters.
- Some string inputs(especially if the user calls an external class from Pytorch), should be double quoted like `"'Parameter Value'"`.
- User may find the information on the support for [Optimizers](#optimizers) later in the documentation.

*Tips for Image Operations (PIL Image Functions)*
- `image_ops` key is responsible for calling PIL operations on the input.
- PIL integration is only supportive for Pytorch Transforms(`transform`, `target_transform` keys) integration. It should be used for secondary purposes. Many of the functions PIL supports is already wrapped in Pytorch Transforms.
- Each dictionary represents a separate operation.
- Some string inputs(especially if user calls an external class from PIL), should be double quoted like `"'Parameter Value'"`
- User may find the information on the support for [Optimizers](#optimizers) later in the documentation.

*Tips for Pytorch Transforms*
- `transform` and `target_transform` keys are both responsible calling Pytorch Transforms. First one is for input, the second one is for label respectively.
- Transforms integration is the main integration responsible for preprocessing images, and labels before training.
- Each key in the dictionary represents a separate operation.
- Order of the keys represent the order of sequential transforms to be applied.
- Transforms without a parameter should be given the value `True` to be passed.
- Some string inputs(especially if the user calls an external class from Pytorch), should be double quoted like `"'Parameter Value'"`
- User may find the information on the support for [Transforms](#transforms) later in the documentation.

*Tips for Label Names*
- `label_names` is responsible for declaring label names.
- Label Names should be present in the Image Database Storage Server created by the user.
- If the user provided `height` and `width` of images to be scraped in `add_to_db` or `multiple_queries` endpoints, the label name should be written with an addendum `imagesize:heightxwidth`. Otherwise the images without certain classification will be fetched if they are present in the server.
- Vectorized versions of labels could be transformed using `target_transform`

*Tips for Model*
- `model` key is responsible for the calling or creation of a model.
- If `name` key is provided, a previously defined class name within `models.py` will be called, and `layers` key will be ignored.
- If `layers` key is provided, and `name` key is not provided, a sequential layer creation will follow.
- Each dictionary in the `layers` array represents a training layer.
- User may use `auto` value for the input parameter to automatically get the past output layer in a limited support. For now, it is only supported for same kinds of layers.
- User may use `n_labels` to indicate the number of labels in the final layer.
- User may find the information on the support for [Layers](#layers) later in the documentation.


# Testing a Model
## `test`
User may test the trained model by fetching random images that have the same classifications as labels.

<details>
<summary>Visual Documentation Playground</summary>

Head to http://localhost:8000/docs#/default/validationtest_test__post and customize the dictionary:
![test](https://user-images.githubusercontent.com/73674035/186639699-b9b58c8e-5cc6-44a5-b2f7-9a94b4708d0a.png)

</details>

<details>
<summary>Curl Command with Explanation of Parameters</summary>

```
curl -X 'POST' \
  'http://localhost:8000/test/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "ids": [
    <Experimental, ids of the specific set of images to be fetched from image database for testing>
  ],
  "limit": <Limit of how many random images with same classification will be fetched from the database>,
  "label_names": [
    <Should be the same label names the user picked for training>
  ],
  "n_labels": 0, ## Should be kept 0, will automatically update in testing process.
  "criterion": {
   <Should be the loss function the user picked for training>
  },
  "model_name": "<Should be the same file name without extension user picked when training>",
  "image_ops": [
    <Should be the image operations the user picked for training>
  ],
  "transform": {
    <Should be the same input transformation the user picked for training>
  },
  "target_transform": {
    <Should be the same label transformation the user picked for training>
  },
  "model": {
    "name": <Should be the class name the user picked for training>,
    "layers": [
      <Should be the same layers the user picked for training>
    ]
  }
}'
```
</details>

### Example Dictionary
```py
{
  "ids": [
  ],
  "limit": 200,
  "label_names": [
    "American Hairless Terrier imagesize:500x500",
    "Alaskan Malamute imagesize:500x500",
    "American Eskimo Dog imagesize:500x500",
    "Australian Shepherd imagesize:500x500",
    "Boston Terrier imagesize:500x500",
    "Boykin Spaniel imagesize:500x500",
    "Chesapeake Bay Retriever imagesize:500x500",
    "Catahoula Leopard Dog imagesize:500x500",
    "Toy Fox Terrier imagesize:500x500"
  ],
  "n_labels": 0,
  "criterion": {
    "name": "CrossEntropyLoss"
  },
  "model_name": "american_dog_species",
  "image_ops": [
    {
      "resize": {
        "size": [
          500,
          500
        ],
        "resample": "Image.ANTIALIAS"
      }
    },
    {
      "convert": {
        "mode": "'RGB'"
      }
    }
  ],
  "transform": {
    "ToTensor": True,
    "Normalize": {
      "mean": [
        0.5,
        0.5,
        0.5
      ],
      "std": [
        0.5,
        0.5,
        0.5
      ]
    }
  },
  "target_transform": {
    "ToTensor": True
  },
  "model": {
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
        "inplace": True
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
        "inplace": True
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
        "inplace": True
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
        "in_features": 111392,
        "out_features": 120
      },
      {
        "name": "ReLU",
        "inplace": True
      },
      {
        "name": "Linear",
        "in_features": "auto",
        "out_features": 84
      },
      {
        "name": "ReLU",
        "inplace": True
      },
      {
        "name": "Linear",
        "in_features": "auto",
        "out_features": "n_labels"
      }
    ]
  }
}
```
---

# Getting Information on the Training and Testing of the Model
## `find_attempt`
Each time a user uses `train` endpoint, an `Attempt` object is created in the database. This object is also updated on each time `test` endpoint is used. Also, user may automatically check the status of the training from this object.

* At the beginning of each training, the `status` of the object will be `Training`.
* At the end of each training, the `status` of the object will be `Trained`
* At the end of each testing, the `status` of the object will be `Complete`

<details>
<summary>Visual Documentation Playground</summary>

Head to http://localhost:8000/docs#/default/find_attempt_find_attempt__post and enter the name of the model(also the filename without extension): 
![find_attempt](https://user-images.githubusercontent.com/73674035/186650147-8a47bdec-2b9b-45a0-a9ae-9ad9cd76ece2.png)
</details>

<details>
<summary>Curl Command with Explanation of Parameters</summary>

```
curl -X 'POST' \
  'http://localhost:8000/find_attempt/?name=<Name of the Model(Also the filename of the saved model without extensions)>' \
  -H 'accept: application/json' \
  -d ''
```

</details>

### Example Output Dictionary

```py
{
  "accuracy": <Accuracy of the Model>,
  "id": <ID of the attempt>,
  "limit": <Limit of the number of testing>,
  "n_epoch": <Number of epochs the model is trained for>,
  "name": "<Name of the Model>",
  "status": "<Status of the Attempt>",
  "testing_commands": {
    <Same as testing commands used in `test` endpoint>
  },
  "training_commands": {
    <Same as tranining commands used in `train` endpoint>
  },
  "training_losses": [
    ## Training Losses for each epoch for observing training quality
    2.1530826091766357,
    2.2155375480651855,
    2.212409019470215,
    ...
  ]
}
```
---

# Support for Various Elements

Below are the different functions, and algorithms supported. Data has been derived from the results of `test_main.py` unit tests. Functions, and algorithms not present in the list may or may not work. Feel free to try them out.

## Layers

<details>
<summary>Supported Pytorch Convolutional Layers</summary>

- Conv1d
  - `dtype` and `device` parameters are not supported.
- Conv2d
  - `dtype` and `device` parameters are not supported.
- Conv3d
  - `dtype` and `device` parameters are not supported.
- ConvTranspose1d
  - `dtype` and `device` parameters are not supported.
- ConvTranspose2d
  - `dtype` and `device` parameters are not supported.
- ConvTranspose3d
  - `dtype` and `device` parameters are not supported.
- LazyConv1d
  - `dtype` and `device` parameters are not supported.
- LazyConv2d
  - `dtype` and `device` parameters are not supported.
- LazyConv3d
  - `dtype` and `device` parameters are not supported.
- LazyConvTranspose1d
  - `dtype` and `device` parameters are not supported.
- LazyConvTranspose2d
  - `dtype` and `device` parameters are not supported.
- LazyConvTranspose3d
  - `dtype` and `device` parameters are not supported.
- Unfold
- Fold
</details>

<details>
<summary>Unsupported Pytorch Convolutional Layers</summary>

None
</details>

<details>
<summary>Supported Pytorch Pooling Layers</summary>

- MaxPool1d
- MaxPool2d
- MaxPool3d
- MaxUnpool1d
- MaxUnpool2d
- MaxUnpool3d
- AvgPool1d
- AvgPool2d
- AvgPool3d
- FractionalMaxPool2d
  - `_random_samples` parameter is not supported.
- FractionalMaxPool3d
  - `_random_samples` parameter is not supported.
- AdaptiveMaxPool1d
- AdaptiveMaxPool2d
- AdaptiveMaxPool3d
- AdaptiveAvgPool1d
- AdaptiveAvgPool2d
- AdaptiveAvgPool3d
</details>

<details>
<summary>Unsupported Pytorch Pooling Layers</summary>

- LPPool1d
- LPPool2d
</details>

<details>
<summary>Supported Pytorch Linear Layers</summary>

- Linear
- Bilinear
- LazyLinear
</details>

<details>
<summary>Unsupported Pytorch Linear Layers</summary>

- Identity

</details>

<details>
<summary>Supported Pytorch Utility Functions From Other Modules</summary>

- Flatten
</details>

<details>
<summary>Unsupported Pytorch Utility Functions From Other Modules</summary>

- Unflatten
</details>

<details>
<summary>Supported Pytorch Non-Linear Activation Layers</summary>

- ELU
- Hardshrink
  - `lambda` parameter is not supported.
- Hardsigmoid
- Hardtanh
  - `min_value` and `max_value` parameters are same as `min_val` and `max_val` respectively.
- Hardswish
- LeakyReLU
- LogSigmoid
- MultiheadAttention
  - `device`, and `dtype` parameters are not supported.
- PReLU
  - `device`, and `dtype` parameters are not supported.
- ReLU
- ReLU6
- RReLU
- SELU
- CELU
- GELU
  - `approximate` parameter is not supported.
- Sigmoid
- SiLU
- Mish
- Softplus
- Softshrink
  - `lambda` parameter is not supported.
- Softsign
- Tanh
- Tanhshrink
- Threshold
- GLU
</details>

<details>
<summary>Unsupported Pytorch Non-Linear Activation Layers</summary>

None
</details>

## Optimizers

<details>
<summary>Supported Pytorch Optimizer Algorithms</summary>

- Adadelta
- Adagrad
- Adam
- AdamW
- Adamax
- ASGD
- NAdam
- RAdam
- RMSprop
- Rprop
- SGD

`foreach`, `maximize`, and `capturable` parameters have been deprecated.
</details>

<details>
<summary>Unsupported Pytorch Optimizer Algorithms</summary>

- LBFGS
</details>

## Loss Functions

<details>
<summary>Supported Pytorch Loss Functions</summary>

- L1Loss
- MSELoss
- CrossEntropyLoss
  - `weight`, and `ignore_index` parameters are not supported yet.
- PoissonNLLLoss
- KLDivLoss
- BCEWithLogitsLoss
  - `weight`, and `pos_weight` parameters are not supported yet.
- HingeEmbeddingLoss
- HuberLoss
- SmoothL1Loss
- SoftMarginLoss
- MultiLabelSoftMarginLoss
  - `weight` parameter is not supported yet.
</details>

<details>
<summary>Unsupported Pytorch Loss Functions</summary>

- CTCLoss
- NLLLoss
- GaussianNLLLoss
- BCELoss
- MarginRankingLoss
- MultiLabelMarginLoss
- CosineEmbeddingLoss
- MultiMarginLoss
- TripletMarginLoss
- TripletMarginWithDistanceLoss
</details>

## Transforms

<details>
<summary>Supported Pytorch Transforms</summary>

- CenterCrop
- ColorJitter
- FiveCrop
- Grayscale
- Pad
- RandomAffine
- RandomCrop
- RandomGrayscale
- RandomHorizontalFlip
- RandomPerspective
- RandomResizedCrop
- RandomRotation
- RandomVerticalFlip
- Resize
- TenCrop
- GaussianBlur
- RandomInvert
- RandomPosterize
- RandomSolarize
- RandomAdjustSharpness
- RandomAutocontrast
- RandomEqualize
- Normalize
- RandomErasing
- ToPILImage
- ToTensor
- PILToTensor
- RandAugment
- TrivialAugmentWide
</details>

<details>
<summary>Unsupported Pytorch Transforms</summary>

- RandomApply
- RandomChoice
- RandomOrder
- LinearTransformation
- ConvertImageDtype
- Lambda
- AutoAugmentPolicy
- AutoAugment
- AugMix
- All Functional Transforms
</details>

## Image Operations

<details>
<summary>Supported Image Operations (Functions from PIL Image Module Image Class)</summary>

- convert
- crop
- effect_spread
- getchannel
- reduce
- resize
- rotate
- transpose
</details>

<details>
<summary>Unsupported Image Operations (Functions from PIL Image Module Image Class)</summary>

- alpha_composite
- apply_transparency
- copy
- draft
- entropy
- filter
- frombytes
- point
- quantize
- remap_palette
- transform
- Any other function that doesn't return an Image object

</details>

---

# Keypoints for the State of the Machine Learning Tool and Its Future Roadmap

- For now, the scope of this software only supports image datasets, and the aim is to create image-classifying machine learning models at scale. The broader purpose is to achieve better computer vision by scalability. Future plans include adding the other basic input tensor types for data science, data analysis, data analytics, or artificial intelligence projects. The open source software could be repurposed to achieve other kinds of tasks such as regression, natural language processing, or any other popular machine learning use cases.

- There are no future plans to support any other programming languages such as Java, Javascript, C/C++, etc. The only supported language will be Python for the foreseeable future. The ability to support other efficient databases on big data such as SQL on Hadoop could be a topic for discussion. Also, the ability to add multiple images from local storage to the storage server is in the future plans.

- The only Machine Learning framework supported is Pytorch. There are plans to extend support for some other machine learning libraries and software such as Tensorflow, Keras, Scikit-Learn, Apache Spark, Scipy, Apache Mahout, Accord.NET, Weka, etc. in the future. Already used libraries such as google-image-results, NumPy, etc. may be utilized further in the future.

- To keep the software user-friendly, the device to train the model on (GPU(CUDA), or CPU) is automatically selected. Also, there are plans to create data visualizations of different models in interactive graphs that can be understood by seasoned data scientists or beginners alike in the future. The drag-and-drop type machine learning software libraries for model creation are not anticipated to be implemented.

- This is open-source software designed for local use. The effects or cost of deployment to cloud servers such as AWS, Google Cloud, etc., or integrating it for machine learning applications with the cloud solutions such as Amazon Sagemaker, IBM Watson, Microsoft’s Azure Machine Learning, and Jupyter Notebook hasn’t been tested yet. Use it at your own discretion. The future plans include some of the large-scale ml tools to be implemented.

- The workflows for future plans above may or may not be implemented depending on the schedule of events, support from other contributors, and its overall use in automation. Multiple machine learning projects with a tutorial will be released explaining machine learning tools.

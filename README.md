# ImageRecognition_Template
Template to start a machine learning task with tensorflow and images labeled as input.

## Requirements

To avoid having to install the development environment on your computer, use docker.

First at all, [install docker on your machine](https://docs.docker.com/engine/installation/) and build the Docker image. For this, you can use the Makefile that is in the docker folder

```
cd ImageRecognition_Template/docker
make image
```
## How to use this code

### To run all unit tests:

```
make test
```

### To train OCR

Download the persistent data folder and place it in resources folder.

https://drive.google.com/file/d/1WApX3xzCdIHR8YIGPTJVl8l8YaHZ3lsn/view?usp=sharing

All commands can be run through a Makefile. For this, launch the following commands from the docker folder : 

```
make annote_ocr
make dataset_ocr
make train_ocr
```

### To train mushroom recognition

Download the persistent data folder and place it in resources folder.

https://drive.google.com/open?id=1LvhkII1W28LYAOY6Kf77bvyzBlBF8VD9

All commands can be run through a Makefile. For this, launch the following commands from the docker folder : 

```
make annote_myco
make dataset_myco
make train_myco
```

It is possible to add and/or remove names of mushrooms to recognize, by modifying the file in resources/mycologie/persists/boletales.txt. Thenceforth training & testing folders should be remove and new dataset will have to be redownloaded before performing other commands.

```
make download_myco
make annote_myco
make dataset_myco
make train_myco
```


### To create your own image dataset & train the model

1) Store images of the training dataset in a hierarchy of folders whose name is the label of the images they contain. For example :

+ data
	+ training
		+ A
			- imgA1.png
			- imgA2.png
			- [...]
		+ B
			- imgB1.png
			- imgB2.png
			- [...]
		+ [...]
	+ testing
		+ A
			- imgA1.png
			- imgA2.png
			- [...]
		+ B
			- imgB1.png
			- imgB2.png
			- [...]
		+ [...]

A MNIST dataset (.png) is available for download : https://www.dropbox.com/s/m05yjxvoydbq1j2/mnist_png.tar.gz?dl=0

Another way to easily create this hierarchy of folders is to use googleimagesdownload : https://github.com/hardikvasa/google-images-download
You will find an example of use in docker/mycologie/download.sh.

2) Generate labelling file : the generated file is a .txt that will associate each image with its label. As input, you must enter the path to the folder containing the hierarchy of data previously created. And as output the path to the txt file to create.

Create a shell script containing the command lines. 

```
#! /bin/bash
cd ..
python -m imageRecognition annote input_training_dir_path output_training_file_path.txt
python -m imageRecognition annote input_testing_dir_path output_testing_file_path.txt
```

Add a rule to the Makefile calling this shell script.

```
annote:
	$(eval WORKING_DIRECTORY = $(DOCKER_WD))
	$(DOCKER) /imageRecognition/docker/annote.sh
```

Run it.

```
make annote
```

3) Generate dataset : the generated file is a TFRecords containing information about the data and their label, in a format easily read by TensorFlow. As input, you must enter the path to the txt previously created. And as output the path to the tfrecords to create.

Create a shell script containing the command lines. 

```
#! /bin/bash
cd ..
python -m imageRecognition dataset input_training_file_path.txt input_training_file_path.tfrecords
python -m imageRecognition dataset input_testing_file_path.txt input_testing_file_path.tfrecords
```

Add a rule to the Makefile calling this shell script.

```
dataset:
	$(eval WORKING_DIRECTORY = $(DOCKER_WD))
	$(DOCKER) /imageRecognition/docker/dataset.sh
```

Run it.

```
make dataset
```

4) Train the model on these data

Create a shell script containing the command lines. 

```
#! /bin/bash
cd ..
python -m imageRecognition train input_training_file_path.tfrecords
```

Add a rule to the Makefile calling this shell script.

```
train:
	$(eval WORKING_DIRECTORY = $(DOCKER_WD))
	$(DOCKER) /imageRecognition/docker/train.sh
```

Run it.

```
make train
```


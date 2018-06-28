# ImageRecognition_Template
Template to start a machine learning task with tensorflow and images labeled as input.

## Requirements

1. Install the TensorFlow library. For example:

```
virtualenv --system-site-packages ~/.tensorflow
source ~/.tensorflow/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow-gpu
```
## How to use this code

Move to the folder containing ImageRecognition_Template.

### To run all unit tests:

```
python -m unittest ImageRecognition_Template.test.tests
```

### To create your own image dataset

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

2) Generate labelling file : the generated file is a .txt that will associate each image with its label. As input, you must enter the path to the folder containing the hierarchy of data previously created. And as output the path to the txt file to create.
```
python -m ImageRecognition_Template annote input_dir_path output_file_path.txt
```

3) Generate dataset : the generated file is a TFRecords containing information about the data and their label, in a format easily read by TensorFlow. As input, you must enter the path to the txt previously created. And as output the path to the tfrecords to create.
```
python -m ImageRecognition_Template dataset intput_file_path.txt output_file_path.tfrecords
```


### To train the model

```
python -m ImageRecognition_Template train input_file_path.tfrecords
```

#! /bin/bash
cd ..
python -m imageRecognition dataset imageRecognition/resources/mycologie/boletales_training.txt imageRecognition/resources/mycologie/boletales_training.tfrecords
python -m imageRecognition dataset imageRecognition/resources/mycologie/boletales_testing.txt imageRecognition/resources/mycologie/boletales_testing.tfrecords
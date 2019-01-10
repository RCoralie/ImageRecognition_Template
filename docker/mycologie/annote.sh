#! /bin/bash
cd ..
python -m imageRecognition annote imageRecognition/resources/mycologie/training imageRecognition/resources/mycologie/boletales_training.txt --rename True --format png
python -m imageRecognition annote imageRecognition/resources/mycologie/testing imageRecognition/resources/mycologie/boletales_testing.txt --rename True --format png
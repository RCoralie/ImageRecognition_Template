#! /bin/bash
cd ..
python -m imageRecognition dataset imageRecognition/resources/ocr/chars_training.txt imageRecognition/resources/ocr/chars_training.tfrecords
python -m imageRecognition dataset imageRecognition/resources/ocr/chars_testing.txt imageRecognition/resources/ocr/chars_testing.tfrecords
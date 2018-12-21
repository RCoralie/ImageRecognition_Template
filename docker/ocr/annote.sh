#! /bin/bash
cd ..
python -m imageRecognition annote imageRecognition/resources/ocr/persists/chars_training imageRecognition/resources/ocr/chars_training.txt
python -m imageRecognition annote imageRecognition/resources/ocr/persists/chars_testing imageRecognition/resources/ocr/chars_testing.txt
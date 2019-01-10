#! /bin/bash
cd ../imageRecognition/resources/mycologie/persists
googleimagesdownload -kf boletales.txt -o ../training -f jpg -ct full-color -s large -t photo -wr '{"time_min":"01/01/2017","time_max":"12/31/2017"}' -l 100
googleimagesdownload -kf boletales.txt -o ../testing -f jpg -ct full-color -s large -t photo -wr '{"time_min":"01/01/2018","time_max":"12/31/2018"}' -l 50
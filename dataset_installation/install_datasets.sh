#!/bin/bash

# navigate to directory to install datasets
cd ~/codes_artifact_datasets

# install DSADS
wget https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip
unzip daily+and+sports+activities.zip -d dsads
rm daily+and+sports+activities.zip

# install PAMAP2
wget https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip
unzip pamap2+physical+activity+monitoring.zip -d pamap2
rm pamap2+physical+activity+monitoring.zip

# install Opportunity
wget https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip
unzip opportunity+activity+recognition.zip -d opportunity
rm opportunity+activity+recognition.zip

# install RWHAR
wget http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip
unzip realworld2016_dataset.zip -d rwhar
rm realworld2016_dataset.zip

# unload pamap2 and rwhar
cp ~/BHAR_codes_artifact/dataset_installation/unload_pamap2.sh .
cp ~/BHAR_codes_artifact/dataset_installation/unload_rwhar.sh .
./unload_pamap2.sh
./unload_rwhar.sh
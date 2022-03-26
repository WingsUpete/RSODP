#!/bin/sh
ulimit -n 65535

### Preprocess ###
cd preprocess/
python DataPackager.py -d ny2016_0101to0331.csv --minLat 40.4944 --maxLat 40.9196 --minLng -74.2655 --maxLng -73.6957 --refGridH 2.5 --refGridW 2.5 -er 0 -od ../data/ -c 20
python DataPackager.py -d dc2017_0101to0331.csv --minLat 38.7919 --maxLat 38.9960 --minLng -77.1200 --maxLng -76.9093--refGridH 2.5 --refGridW 2.5 -er 0 -od ../data/ -c 20
cd ../

### GallatExt ###
python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -net GallatExt
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m eval -e eval.pth



##### Baselines #####
### HA ###
#python HistoricalAverage.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -sch all
#python HistoricalAverage.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -sch tendency
#python HistoricalAverage.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -sch periodicity

### AR ###
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -net AR
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m eval -e eval.pth



##### Other Models #####
### Gallat ###
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -tt pretrain -net Gallat -t 0
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -tt retrain -r pretrained_model.pth -t 0
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m eval -e eval.pth

### Gallat+ ###
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -tt pretrain -net Gallat -t 1
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -tt retrain -r pretrained_model.pth -t 1
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m eval -e eval.pth

### LSTNet ###
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -net LSTNet -rar refAR.pth
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m eval -e eval.pth

### GCRN ###
#python Trainer.py -dr data/ny2016_0101to0331/ -mfb 1 -c 20 -gid 0 -m train -net GCRN
#python Trainer.py -dr data/ny2016_0101to0331/ -mfb 1 -c 20 -gid 0 -m eval -e eval.pth

### GEML ###
#python Trainer.py -dr data/ny2016_0101to0331/ -mfb 1 -c 20 -gid 0 -m train -net GEML
#python Trainer.py -dr data/ny2016_0101to0331/ -mfb 1 -c 20 -gid 0 -m eval -e eval.pth



##### Variants #####
### GallatExt-1 (No tuning) ###
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -net GallatExt -t 0
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m eval -e eval.pth

### GallatExt-2 (Concatenation scheme) ###
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -net GallatExtFull
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m eval -e eval.pth

### GallatExt-3 (Leverage tuning) ###
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -net GallatExt -re 0.2
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m eval -e eval.pth -re 0.2

### GallatExt-4 (Unified graph) ###
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -net GallatExt -u 1
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m eval -e eval.pth -u 1

### GallatExt-5 (Shifting scheme) ###
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 1 -m train -net GallatExt -re -2
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m eval -e model_save/20211201_18_08_18.pth -re -2

### GallatExt-6 (Tuning with AR) ###
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m train -net GallatExt -rar refAR.pth
#python Trainer.py -dr data/ny2016_0101to0331/ -c 20 -gid 0 -m eval -e eval.pth

#!/bin/sh
ulimit -n 65535

### Preprocess ###
cd preprocess/
python DataPackager.py -d ny2016_0101to0331.csv --minLat 40.4944 --maxLat 40.9196 --minLng -74.2655 --maxLng -73.6957 --refGridH 2.5 --refGridW 2.5 -er 0 -od ../data/ -c 20
cd ../

### GallatExt ###
python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m train -tt pretrain -net GallatExt
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m train -tt retrain -r pretrained_model.pth
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m eval -e eval.pth



##### Baselines #####
### HA ###
python HistoricalAverage.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -sch all
python HistoricalAverage.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -sch tendency
python HistoricalAverage.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -sch periodicity

### AR ###
python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m train -net AR



##### Other Models #####
### Gallat ###
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m train -tt pretrain -net Gallat -t 0
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m train -tt retrain -r pretrained_model.pth -t 0
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m eval -e eval.pth

### Gallat+ ###
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m train -tt pretrain -net Gallat -t 1
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m train -tt retrain -r pretrained_model.pth -t 1
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m eval -e eval.pth



##### Variants #####
### GallatExt-1 ###
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m train -tt pretrain -net GallatExt -t 0
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m train -tt retrain -r pretrained_model.pth -t 0
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m eval -e eval.pth

### GallatExt-2 ###
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m train -tt pretrain -net GallatExtFull
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m train -tt retrain -r pretrained_model.pth
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -gid 0 -m eval -e eval.pth

#!/bin/sh
ulimit -n 65535
# Preprocess
cd preprocess/
python DataPackager.py -d ny2016_0101to0331.csv --minLat 40.4944 --maxLat 40.9196 --minLng -74.2655 --maxLng -73.6957 --refGridH 2.5 --refGridW 2.5 -er 0 -od ../data/ -c 20
cd ../
# HA
python HistoricalAverage.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -bs 32 -sch all
python HistoricalAverage.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -bs 32 -sch tendency
python HistoricalAverage.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -bs 32 -sch periodicity
# GallatExt
python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -m train -tt pretrain -net GallatExt -me 300 -bs 32 -re 0.2
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -m train -tt retrain -r model_save/20210528_11_57_39.pth -me 300 -bs 32 -re 0.2
#python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -m eval -e model_save/20210530_05_30_19.pth -bs 32

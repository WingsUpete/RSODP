#!/bin/sh
ulimit -n 65535
# HA
python HistoricalAverage.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -bs 32
# GallatExt
python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -m train -tt pretrain -net GallatExt -me 200 -bs 32 -re 0.2
# python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -m train -tt retrain -r model_save/20210528_11_57_39.pth -me 200 -bs 32 -re 0.2
# python Trainer.py -dr data/ny2016_0101to0331/ -th 2184 -ts 1 -c 20 -m eval -e model_save/20210530_05_30_19.pth -bs 32

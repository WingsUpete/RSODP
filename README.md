# RSODP
Origin-Destination Prediction for Ridesharing System



## I. Introduction

This repository is created for an undergraduate final project which studies the prediction models for Ridesharing use cases. Ridesharing is a blooming service in which passengers share their rides for a variety of purposes (fewer route costs for vehicles, better experiences for passengers, etc). In recent years, researchers have focused largely on traffic forecasting and predictions in order to benefit transportation planning as well as environmental protection. In the case of ridesharing, the most popular topic is Origin-Destination (OD) prediction which intends to predict future passenger demands for better vehicle assignment. In this repository, we will research on the latest designs of OD prediction model for ridesharing system and try to learn the common practice among all for future extension.



## II. Problem Instance

Given a request sequence <img src="https://render.githubusercontent.com/render/math?math=R = [r_0, r_1, \dots, r_{n - 1}]"> of <img src="https://render.githubusercontent.com/render/math?math=n"> requests, where one certain request <img src="https://render.githubusercontent.com/render/math?math=r_i = (t_{r_i}, v_{s_i}, v_{d_i}, n_i)"> stores the request time <img src="https://render.githubusercontent.com/render/math?math=t_{r_i}">, the source and destination coordinates <img src="https://render.githubusercontent.com/render/math?math=v_{s_i} = (lng_{s_i}, lat_{s_i})">, <img src="https://render.githubusercontent.com/render/math?math=v_{d_i} = (lng_{d_i}, lat_{d_i})">, the volume (i.e., number of passengers) of the request <img src="https://render.githubusercontent.com/render/math?math=n_i">, RSODP intends to predict the future requests <img src="https://render.githubusercontent.com/render/math?math=\hat{R}">.



## III. Data

-   [New York Yellow Taxi Trip Data (2016)](https://www.kaggle.com/vishnurapps/newyork-taxi-demand)

-   [Peru Uber Dataset (2010)](https://www.kaggle.com/marcusrb/uber-peru-dataset)
-   [Washington DC Taxi Trips (2017)](https://www.kaggle.com/bvc5283/dc-taxi-trips)

The data has been preprocessed to keep only the information we need. The format of data file is as follow:

```pseudocode
% [DATA TITLE]
% #, request time, src lng, src lat, dst lng, dst lat, volume
0, 2021-03-15 07:15, -73.9767456054687, 40.7698135375976, -74.0042648315429, 40.7461280822753, 2
[...]
```



## IV. Model

…



## V. Experiment

…
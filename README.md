# RSODP
Origin-Destination Prediction for Ridesharing

<br>

## I. Introduction

This repository is created for an undergraduate final project which studies the prediction models for Ridesharing use cases. Ridesharing is a blooming service in which passengers share their rides for a variety of purposes (fewer route costs for vehicles, better experiences for passengers, etc). In recent years, researchers have focused largely on traffic forecasting and predictions in order to benefit transportation planning as well as environmental protection. In the case of ridesharing, the most popular topic is Origin-Destination (OD) prediction which intends to predict future passenger demands for better vehicle assignment. In this repository, we will research on the latest designs of OD prediction model for ridesharing system and try to learn the common practice among all for extension. We further extend [Gallat](https://arxiv.org/pdf/2101.00752.pdf) as GallatExt for better performance on solving RSODP.

<br>

## II. Problem Instance

Given a request sequence <img src="https://render.githubusercontent.com/render/math?math=R = [r_0, r_1, \dots, r_{n - 1}]"> of <img src="https://render.githubusercontent.com/render/math?math=n"> requests, where one certain request <img src="https://render.githubusercontent.com/render/math?math=r_i = (t_{r_i}, v_{s_i}, v_{d_i}, n_i)"> stores the request time <img src="https://render.githubusercontent.com/render/math?math=t_{r_i}">, the source and destination coordinates <img src="https://render.githubusercontent.com/render/math?math=v_{s_i} = (lat_{s_i}, lng_{s_i})">, <img src="https://render.githubusercontent.com/render/math?math=v_{d_i} = (lat_{d_i}, lng_{d_i})">, the volume (i.e., number of passengers) of the request <img src="https://render.githubusercontent.com/render/math?math=n_i">, RSODP intends to predict the future requests <img src="https://render.githubusercontent.com/render/math?math=\hat{R}">.

<br>

## III. Data

-   [New York Yellow Taxi Trip Data (2016)](https://www.kaggle.com/vishnurapps/newyork-taxi-demand)

-   [Peru Uber Dataset (2010)](https://www.kaggle.com/marcusrb/uber-peru-dataset)
-   [Washington DC Taxi Trips (2017)](https://www.kaggle.com/bvc5283/dc-taxi-trips)

The data has been preprocessed to keep only the information we need. The format of data file is as follow:

```pseudocode
[request time], [src lat], [src lng], [dst lat], [dst lng], [volume]
2021-03-15 07:15:00, 40.7698135375976, -73.9767456054687, 40.7461280822753, -74.0042648315429, 2
[...]
```

<br>

## IV. Model

Our model is named as **GallatExt** since it is an extension version of [Gallat](https://arxiv.org/pdf/2101.00752.pdf). Other models for comparison are listed below.

### Baseline

#### <img src="https://render.githubusercontent.com/render/math?math=HA^{%2b}"> (Historical Average)

Historical Average is the very baseline method which computes the average of the historical demands from the previous time slots. For improvement, we further consider the four temporal aspects in our Temporal Attention Layer design while calculate the average of all these values directly.

#### Gallat

Our model is extended from [Gallat](https://arxiv.org/pdf/2101.00752.pdf) (Graph prediction with all attention), so it is considered as an important baseline model for comparison.

#### <img src="https://render.githubusercontent.com/render/math?math=Gallat^{%2b}">

We combine the original Gallat with our idea of fine-tuning in the Transferring Layer since the original Gallat does not appear to achieve better results than the baseline HA.

### Variant

#### GallatExt-1

This version inherits the design of the transferring layer in Gallat, meaning there is no tuning blocks with referenced HA results.

#### GallatExt-2

This version uses concatenation as the aggregation scheme in both spatial and temporal layer (Default aggregation scheme for GallatExt is average).

<br>

## V. Experiment

### Metrics

For most papers focusing on this area, there are three classic metrics for evaluation generally, i.e., the Rooted Mean Square Error (RMSE), Mean Average Percentage Error (MAPE) and Mean Absolute Error (MAE). Their formulas are as follows:

<p align="center"><img src="https://render.githubusercontent.com/render/math?math=RMSE(y, \hat{y}) = \sqrt{\frac{1}{z}\sum_{i=1}^{z}(y - \hat{y})^2}"></p>

<p align="center"><img src="https://render.githubusercontent.com/render/math?math=MAPE(y, \hat{y}) = \frac{1}{z}\sum_{i=1}^{z}|\frac{y - \hat{y}}{y + 1}|"></p>

<p align="center"><img src="https://render.githubusercontent.com/render/math?math=MAE(y, \hat{y}) = \frac{1}{z}\sum_{i=1}^{z}|y - \hat{y}|"></p>

In these formulas, <img src="https://render.githubusercontent.com/render/math?math=z"> represents the number of samples. <img src="https://render.githubusercontent.com/render/math?math=y_i"> and <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> represent the ground truth value and the predicted value respectively.


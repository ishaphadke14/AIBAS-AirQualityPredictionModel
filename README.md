# Air Quality Prediction Model

This repository contains a machine learning model that predicts air quality based on historial pollutant values. It was developed as part of the AIBAS coursework project.

## Project Overview

This project focuses on predicting Air Quality Index(AQI) using supervised machine learning approach. Artificial Neural Network(ANN) and Ordinary Least Squares(OLS)  Models are trained on historical air quality data scraped from the internet. Model Performance is evaluated using regression metrics and visual analysis of predictions.

##  Features

- Data Scraping
- Data Cleaning and Preprocessing
- Feature Engineering
- Model Training using Regression Algorithms
- Model Evaluation (R², RMSE)
- Docker Images

##  Dataset
The Data was obtained from the following sources:

  https://www.accuweather.com

  https://aqicn.org

The dataset includes listings with the following features:
- `pm25`
- `pm10`
- `co`
- `no2`
- `o3`
- `so2`
- `timestamp`

## Technologies Used

- Python 3.x
- Dockerfile
- Matplotlib, Seaborn
- Scikit-learn
- Pandas, NumPy

## Getting Started

1. Clone the repository
  ```bash
  git clone https://github.com/ishaphadke14/AIBAS-AirQualityPredictionModel
  cd AIBAS-AirQualityPredictionModel
  ```

##  Authors

Isha Phadke (ishaphadke14@gmail.com)   Rutuja Argade (rutujaargade4444@gmail.com)

Universität Potsdam

##  License

This project is licensed under the MIT License.
See the [LICENSE](./LICENSE) file for details.
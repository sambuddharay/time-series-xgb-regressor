# Time Series XGB Regressor
Forecasting energy/electricity demand with XGB Regressor  

[Notebook](https://github.com/sambuddharay/time-series-xgb-regressor/blob/main/Time_series_with_XGBoost.ipynb)  
[Data](https://github.com/sambuddharay/time-series-xgb-regressor/blob/main/PJME_hourly.csv)  
[Additional Python package](https://github.com/sambuddharay/time-series-xgb-regressor/blob/main/utils.py)

# Motivation
The goal of this project is to apply gradient boosting to forecast problems with specific considerations and modifications. Time-series data has a temporal structure, and this project aims to predict future values based on historical observations.

Load forecasting is a technique used by the energy-providing utility to predict the electrical power needed to meet the demand and supply equilibrium. The technique can provide a reference for the daily operation of regional power grids and the formulation of dispatching plans. According to the results of power load forecasting, dispatchers can reasonably coordinate the distribution of the output of each power plant, maintain a balance between supply and demand, and ensure power grid stability. This determines the start-stop arrangement of the generator set, reduces the redundant generator reserve capacity value, and reduces the power generation cost.

Even a small improvement in such a demand forecasting system can help save a large amount in expenses in term of workforce management, inventory cost and out of stock loss, which is what is being attempted herein by treating it as a regression problem and using tree-based algorithms to solve it.

# Approach
The correlation between features like days, months, hours, etc. with the target is visualised with the help of graphs. The data is split into training and testing data (which is also visualised with the help of a graph). Some new features are generated (feature engineering) from the given data. An XGBoost regression model is then trained on the training data. After being fed the features of the testing data, its predictions are compared with the actual target values with the help of visualisations and error function.

# References
[https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel](https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel)
[https://github.com/panambY/Hourly_Energy_Consumption](https://github.com/panambY/Hourly_Energy_Consumption)  
[https://github.com/rmhachar/time-series-forecasting-methods](https://github.com/rmhachar/time-series-forecasting-methods)

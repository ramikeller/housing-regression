# Data

## housing.csv

**Source:** StatLib repository, originally from the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron.

**Download URL:** https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv

**Description:** California housing data from the 1990 US Census. Each row represents one census block group.

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| longitude | float | East-west position of the block |
| latitude | float | North-south position of the block |
| housing_median_age | float | Median age of houses in the block |
| total_rooms | float | Total number of rooms across all houses |
| total_bedrooms | float | Total number of bedrooms (some missing values) |
| population | float | Number of people living in the block |
| households | float | Number of households in the block |
| median_income | float | Median household income (tens of thousands of USD) |
| median_house_value | float | Median house value in USD (prediction target) |
| ocean_proximity | string | Categorical: distance to the ocean (not used in current model) |

**Rows:** 20,640 (207 dropped due to missing `total_bedrooms`)

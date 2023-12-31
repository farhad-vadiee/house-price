
# Project: Housing Price Prediction

## Overview
This project involves predicting housing prices using various features and machine learning models. The primary dataset used is the 'kc_house_data_NaN.csv', which includes multiple features related to housing properties.

## Installation
To run this project, ensure you have the following Python libraries installed:
- Pandas
- Matplotlib
- Numpy
- Seaborn
- Scikit-learn

Use the following command to install these libraries:
```bash
pip install pandas matplotlib numpy seaborn scikit-learn
```

## Dataset
The dataset 'kc_house_data_NaN.csv' is loaded and preprocessed. The preprocessing steps include:
- Reading the dataset using Pandas.
- Dropping unnecessary columns.
- Handling missing values in 'bedrooms' and 'bathrooms' columns.
- Generating descriptive statistics of the data.

## Data Visualization
Several plots are used to visualize the data:
- Boxplot to observe the distribution of prices by waterfront view.
- Regression plot to examine the relationship between square footage and price.

## Models
Multiple linear regression models are developed:
1. **Linear Regression**: To predict housing prices using various features.
2. **Ridge Regression**: To address multicollinearity in linear regression.
3. **Polynomial Regression**: To capture non-linear relationships between features and price.

## Pipeline
A pipeline is created combining scaling, polynomial feature transformation, and linear regression to streamline the process of data transformations and model fitting.

## Cross-Validation and Model Evaluation
- The dataset is split into training and testing sets to evaluate model performance.
- Cross-validation scores are used to assess the effectiveness of the models.

## Usage
Run the Python script to load the data, train the models, and evaluate their performance. The script will print out various statistics, model scores, and display plots for data analysis.

## Conclusion
The project demonstrates the application of different regression techniques to predict housing prices. It provides insights into how different features influence house prices and the effectiveness of each model in this prediction task.

## Contribution
Contributions to improve the models, add new features, or enhance data visualization are welcome. Please follow coding standards and document changes.

## License
This project is open-source and available under the [MIT License](LICENSE.txt).

---

*Note: This project is for educational purposes. Real-world datasets and more complex models may be needed for accurate price predictions.*

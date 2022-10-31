
# Premium Product Predictor (In Progress)

## Introduction
The primary goal of this study is to predict whether a manufactured product is premium or non-premium. 

The manufacturer can only identify whether a product is premium or non-premium after its commercialization in the market. They consider that predicting whether a product is a premium or not can affect the planning, storage, and logistic processes of that product. As a result, it can lead to cost and expense optimization. 

Therefore, the company's primary concern is to minimize the possible incorrect prediction of premium products. 

## Dataset Description
_Disclaimer: Due to confidentiality reasons, it is not possible to present or divulge all aspects of this dataset._

This dataset was provided by a manufacturer. It contains information about a manufactured product that can be classified as premium or non-premium. For the company, the term "premium" defines a product with specific prominent characteristics and, consequently, a superior margin. The margin of a premium product is 0.50 USD, while for a non-premium one, the margin is just 0.10 USD.

There are 3 types of predictors: inputs, time, and temperature. The outcome variable is categorical and indicates whether a product batch is premium (1) or not (0). There are much more observations for non-premium than for premium products, which means this is an imbalanced dataset with a ratio of 7.2 non-premium products for each premium product.

### Summary of changes 
* Rearranged columns so that the target variable **column_33** goes first. 
* Calculated missing values for all features. 
* Checked duplicate values for all features. 
* Found the unique values for all features to transform them from float to integer in the applicable cases.
* Dropped **column_35** and **column_72**, the former due to data leakage, and the latter is not considered relevant according to domain experts. 
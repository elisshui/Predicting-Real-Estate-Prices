<h1 align="center">Predicting Real Estate Prices (MATLAB)</h1>
Predicting real estate prices through deploying a linear regressing model using MATLAB - an accuracy of 82.258% was achieved.

#Part 1: Loading and Preprocessing the Data
Loading realestate.csv into a table named dataset.
```matlab
% This line ensures the matrix values are displayed properly.
format short g
```
Loading realestate.csv into a table named dataset.
```
dataset = readtable('realestate.csv')
```
The dataset has 414 data points (rows). 
Taking a random sample of 70% of the data (290 rows) for the training set and leaving the rest for the testing set.
* Shuffle the table rows using randperm.
* Store the first 290 rows in your training set in data_train
* Store the remainder of the rows in data_test
```
dataset = dataset(randperm(height(dataset)), :);

% Spliting the dataset.
data_train = dataset(1:290,:)
data_test = dataset(291:414,:)
```
#Part 2: Linear Regression

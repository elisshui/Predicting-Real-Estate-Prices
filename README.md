<h1 align="center">Predicting Real Estate Prices (MATLAB)</h1>
Predicting real estate prices through deploying a linear regressing model using MATLAB - an accuracy of 82.258% was achieved. This project was the final assessment for the MATLAB component of a university linear algebra course. The goal of this project was to gain experience and assess skills in MATLAB and machine learning.

<h2 align="left">Part 1: Loading and Preprocessing the Data</h2>
Loading realestate.csv into a table named dataset.
```matlab
% This line ensures the matrix values are displayed properly.
format short g
```
Loading realestate.csv into a table named dataset.
```matlab
dataset = readtable('realestate.csv')
```
The dataset has 414 data points (rows). 
Taking a random sample of 70% of the data (290 rows) for the training set and leaving the rest for the testing set.
* Shuffle the table rows using randperm.
* Store the first 290 rows in the training set in data_train.
* Store the remainder of the rows in data_test.
```matlab
dataset = dataset(randperm(height(dataset)), :);

% Spliting the dataset.
data_train = dataset(1:290,:)
data_test = dataset(291:414,:)
```
<h2 align="left">Part 2: Linear Regression</h2>
Finding a linear regression model to predict the price per unit area of the house based on three housing features: house age, distance to the nearest MRT station, and number of convenient stores.
```matlab
% Building up the linear system of equations and finding the coefficient of
% the training data.
yTrain = data_train.House_Price_of_Unit_Area;
X1Train = data_train.House_Age;
X2Train = data_train.Distance_to_Nearest_MRT_station;
X3Train = data_train.Number_of_Convenience_Stores;
NTrain = 290

ATrain = zeros(4);
CTrain = [];

% Building matrix A
ATrain(1,1) = NTrain;
ATrain(1,2) = sum(X1Train);
ATrain(1,3) = sum(X2Train);
ATrain(1,4) = sum(X3Train);
ATrain(2,1) = sum(X1Train);
ATrain(2,2) = sum(X1Train.^2);
ATrain(2,3) = sum(X1Train.*X2Train);
ATrain(2,4) = sum(X1Train.*X3Train);
ATrain(3,1) = sum(X2Train);
ATrain(3,2) = sum(X2Train.*X1Train);
ATrain(3,3) = sum(X2Train.^2);
ATrain(3,4) = sum(X2Train.*X3Train);
ATrain(4,1) = sum(X3Train);
ATrain(4,2) = sum(X3Train.*X1Train);
ATrain(4,3) = sum(X3Train.*X2Train);
ATrain(4,4) = sum(X3Train.^2)

% Building matrix C
CTrain(1,1) = sum(yTrain);
CTrain(2,1) = sum(X1Train.*yTrain);
CTrain(3,1) = sum(X2Train.*yTrain);
CTrain(4,1) = sum(X3Train.*yTrain)

% Solving for ß
BTrain = inv(ATrain)*CTrain
```
Root mean square error (RMSE) is one of the metrics used to evaluate the performance of the linear regression model. Thus, the RMSE of the train and test set models were found.
```matlab
% Train set
yiTrain = BTrain(1)+ BTrain(2)*X1Train + BTrain(3)*X2Train + BTrain(4)*X3Train;
RMSE_train = sqrt(sum((yiTrain-yTrain).^2)/NTrain)

% Test set
yTest = data_test.House_Price_of_Unit_Area;
X1Test = data_test.House_Age;
X2Test = data_test.Distance_to_Nearest_MRT_station;
X3Test = data_test.Number_of_Convenience_Stores;
NTest = 414-NTrain;

ATest = zeros(4);
CTest = [];
% Building matrix A
ATest(1,1) = NTest;
ATest(1,2) = sum(X1Test);
ATest(1,3) = sum(X2Test);
ATest(1,4) = sum(X3Test);
ATest(2,1) = sum(X1Test);
ATest(2,2) = sum(X1Test.^2);
ATest(2,3) = sum(X1Test.*X2Test);
ATest(2,4) = sum(X1Test.*X3Test);
ATest(3,1) = sum(X2Test);
ATest(3,2) = sum(X2Test.*X1Test);
ATest(3,3) = sum(X2Test.^2);
ATest(3,4) = sum(X2Test.*X3Test);
ATest(4,1) = sum(X3Test);
ATest(4,2) = sum(X3Test.*X1Test);
ATest(4,3) = sum(X3Test.*X2Test);
ATest(4,4) = sum(X3Test.^2)

% Building matrix C
CTest(1,1) = sum(yTest);
CTest(2,1) = sum(X1Test.*yTest);
CTest(3,1) = sum(X2Test.*yTest);
CTest(4,1) = sum(X3Test.*yTest)

% Solving for ß
Btest = inv(ATest)*CTest 

yiTest = Btest(1)+ Btest(2)*X1Test + Btest(3)*X2Test + Btest(4)*X3Test;
RMSE_test = sqrt(sum((yiTest-yTest).^2)/NTest)
```
<h2 align="left">Part 3: Classification</h2>
Building a classification model to predict whether houses will be worth below 40 dollars per unit area. A linear regression model will be applied on the train set of data and it's performance will be tested on the test data set.

A copy of the data set used above is used so as not to risk using corrupt data.
```matlab
% Creating a copy of the dataset.
CopyDataSet = dataset

% Changing the price per unit area to 0 if it's less than or equal to 40 and changing it to 1 if it's more than 40.
CopyDataSet.House_Price_of_Unit_Area(CopyDataSet.House_Price_of_Unit_Area <= 40) = 0
CopyDataSet.House_Price_of_Unit_Area(CopyDataSet.House_Price_of_Unit_Area > 40) = 1

% Spliting the data in to train set and test set, with 70% of data going to train set.
CopyDataSet = CopyDataSet(randperm(height(CopyDataSet)), :);

% 70 percent of 414 is 289.8. Thus, 290 goes into the training set.
data_train = CopyDataSet(1:290,:);
data_test = CopyDataSet(291:414,:);

yCopyTRAIN = data_train.House_Price_of_Unit_Area
yCopyTEST = data_test.House_Price_of_Unit_Area
```
Training the model on the modified train set.
```matlab
% Setting up the a1, a2, a3 array and the error matrix.
% Note: It's is known that the optimal value for  is 2.1. This section serves to search for optimal values for a1 between -0.07
% and -0.05, a2 between -0.003 and -0.001 and a3 between 0.1 and 0.3, each with a step-space of 40.

Error = []
a0 = 2.1
a1 = linspace(-0.07,-0.05, 40)
a2 = linspace(-0.003, -0.001, 40)
a3 = linspace(0.1, 0.3, 40)

X1 = data_train.House_Age
X2 = data_train.Distance_to_Nearest_MRT_station
X3 = data_train.Number_of_Convenience_Stores
```
```matlab
% Calculating the error for all combinations use for loops using the "mysigmoid" function created in the function section (Part 4).
for i = 1:1:40
    for j = 1:1:40
        for k = 1:1:40
            X = (2.1+a1(i)*X1(i)+a2(j)*X2(j)+a3(k)*X3(k));
            Error(i,j,k) = mysigmoid(X);
        end
    end
end
```
```matlab
% Finding the optimal value for a1, a2, and a3.
[minError,loc] = min(Error(:))
[i,j,k]= ind2sub(size(Error), loc)

optimal_a1 = a1(i)
optimal_a2 = a2(j)
optimal_a3 = a3(k)
```
If the predicted value is less than or equal 0.5, it will be considered as 0 (lower than 40), otherwise it will be considered as 1 (higher than 40).
```matlab
% Calculating the accuracy of the model on the train set.
yhat = mysigmoid(a0+a1(i)*X1+a2(j)*X2+a3(k)*X3);
yhat(yhat <= 0.5) = 0;
yhat(yhat > 0.5) = 1;

accTraiN = myfunc_accuracy(yhat,yCopyTRAIN)

% Calculating the accuracy of the model on the test set.
test1 = data_test.House_Age;
test2 = data_test.Distance_to_Nearest_MRT_station;
test3 = data_test.Number_of_Convenience_Stores;
testy = data_test.House_Price_of_Unit_Area;

yhat = mysigmoid(a0+a1(i)*test1+a2(j)*test2+a3(k)*test3);
yhat(yhat <= 0.5) = 0;
yhat(yhat > 0.5) = 1;

accTesT = func_accuracy(yhat,yCopyTEST)
```

<h2 align="left">Part 4: Functions Used</h2>
```matlab
function f = mysigmoid(x)
    f = 1./(1+exp(-x));
end

function accTrain = myfunc_accuracy(yhat,yCopyTRAIN)
equalYs= (yhat == yCopyTRAIN);
accTrain = sum(equalYs/length(equalYs))*100;
end

function accTest = func_accuracy(yhat,yCopyTEST)
equalYs= (yhat == yCopyTEST);
accTest = sum(equalYs/length(equalYs))*100;
end
```
---

Project by [Eliss Hui](https://github.com/elisshui "Eliss Hui") (Nov 2021)

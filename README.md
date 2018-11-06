# Data-Science-Projects

A collection of beginner data science projects using open source data to learn and have fun. 

1. **House Prices (regression)**

In this project, the goal is to use house price data (availabe through [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) from homes in Ames, Iowa) to create a model that will help predict house prices based on an original 79 house features.

Four regression models are built: 1) random forest; 2) ridge regression; 3) lasso regression; and 4) elastic net using k-fold cross validation with 5 samples. Random forest was the best performer, generating the lowest median root mean square error of all models across resamples.

2. **Human Activity Recognition (classification)** [HAR github page](https://msra7rr2.github.io/Data-Science-Projects/index.html)

In this project, I used data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to build a model that accurately predicts *how well* they did a weight lifting exercise (categorical 'classe' variable).

A classification tree is built that yields very low accuracy (37%), compared to a random forest model that has 98% accuracy. 

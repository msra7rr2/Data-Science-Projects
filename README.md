# Data-Science-Projects

A collection of beginner data science projects using open source data to learn and have fun. 

1. **House Prices (regression)** [HP github page](https://msra7rr2.github.io/Data-Science-Projects/House-Prices/index_housing_prices.html)

In this project, the goal is to use house price data (availabe through [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) from homes in Ames, Iowa) to create a model that will help predict house prices based on an original 79 house features.

Four regression models are built: 1) random forest; 2) ridge regression; 3) lasso regression; and 4) elastic net using k-fold cross validation with 5 samples. Random forest was the best performer, generating the lowest median root mean square error of all models across resamples.

2. **Human Activity Recognition (classification)** [HAR github page](https://msra7rr2.github.io/Data-Science-Projects/Human-Activity-Recognition/index.html)

In this project, I used data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to build a model that accurately predicts *how well* they did a weight lifting exercise (categorical 'classe' variable).

A classification tree is built that yields very low accuracy (37%), compared to a random forest model that has 98% accuracy.


3. **Coursework assignments** 

+**Data Cleaning** [link]

Write code that does the following and submit tidy dataset: 
1. Merges the training and the test sets to create one data set.
2. Extracts only the measurements on the mean and standard deviation for each measurement.
3. Uses descriptive activity names to name the activities in the data set
4. Appropriately labels the data set with descriptive variable names.
5. From the data set in step 4, creates a second, independent tidy data set with the average of each variable for each activity and each subject.

+**Statistical Inference** [link]

+Part 1: Simulation exercise/Central Limit Theorem: We will be investigating the exponential distribution in R and comparing it with the Central Limit Theorem by analysing the distribution of averages of 40 exponentials, and doing 1,000 simulations.

+Part 2: Hypothesis testing/Differences in tooth length

+**Regression** [link]

Brief exercise using linear regression to explore MPG (Miles/(US) gallon) and Transmission in the mtcars dataset to determine whether 1) an automatic or manual transmission better for MPG; and to 2) quantify the MPG difference between automatic & manual transmissions.



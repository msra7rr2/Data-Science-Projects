library(tidyverse)
library(stringr)
library(dplyr)
library(tidyr)
library(data.table)
library(xlsx)
library(data.table)
library(lubridate)
options(scipen=999)

setwd("/Users/raqdom/DataScience_Coursera/03_Data_Cleaning")

#1. Merges the training and the test sets to create one data set.
#2. Extracts only the measurements on the mean and standard deviation for each measurement.
#3. Uses descriptive activity names to name the activities in the data set
#4. Appropriately labels the data set with descriptive variable names.
#5. From the data set in step 4, creates a second, independent tidy data set with the average of each variable for each activity and each subject.

######## STEPS 1, 3 & 4  ########

  #importing 'train' files (inc set file, activity file, subject id file)
  train_set <- tbl_df(read.table("./UCI HAR Dataset/train/X_train.txt", quote="\"", comment.char=""))
  train_activity_labels <- tbl_df(read.table("./UCI HAR Dataset/train/y_train.txt", quote="\"", comment.char=""))
  train_subject_id <- tbl_df(read.table("./UCI HAR Dataset/train/subject_train.txt", quote="\"", comment.char=""))

  #importing 'test' files (inc set file, activity file, subject id file)
  test_set <- tbl_df(read.table("./UCI HAR Dataset/test/X_test.txt", quote="\"", comment.char=""))
  test_activity_labels <- tbl_df(read.table("./UCI HAR Dataset/test/y_test.txt", quote="\"", comment.char=""))
  test_subject_id <- tbl_df(read.table("./UCI HAR Dataset/test/subject_test.txt", quote="\"", comment.char=""))

  #importing feature & activity labels for both train/test
  features <- read.table("./UCI HAR Dataset/features.txt")
  activities <- read.table("./UCI HAR Dataset/activity_labels.txt")
  
  #before merging, adding DESCRIPTIVE COLUMN NAMES from features file (make.names to get rid of invalid characters)
  #adding DESCRIPTIVE ACTIVITY NAMES to both train/test 
  
  colnames(test_set) <- make.names(features$V2, unique = TRUE, allow_ = TRUE) #feature columns
  colnames(train_set) <- make.names(features$V2, unique = TRUE, allow_ = TRUE)

  train_activity_labels <- train_activity_labels %>% #activities 
    left_join(activities) %>%
    rename(activity = "V2") %>%
    select(-V1)
  
  test_activity_labels <- test_activity_labels %>% 
    left_join(activities) %>%
    rename(activity = "V2") %>%
    select(-V1)
  
  train_subject_id <- rename(train_subject_id, subjectid = V1) #other descriptive column names
  test_subject_id <- rename(test_subject_id, subjectid = V1)
  
  #completing test & train sets before merging (binding data, activities, subjects)
  
  train_full <- bind_cols(train_set, train_subject_id, train_activity_labels)
  test_full <-  bind_cols(test_set, test_subject_id, test_activity_labels)

  #MERGING into 1 full dataset
  
  full <- bind_rows(train_full, test_full)
  
######## STEP 2 : Extracts only the measurements on the mean and standard deviation for each measurement ########
  
  full <- full %>% 
    select (contains("subjectid"), contains("activity"), contains(".mean."), contains(".std.")) 
                            
######## STEP 5 : From the data set in step 4, creates a second, independent tidy data set with the average of each variable 
#for each activity and each subject ########

tidy_average_data <- full %>%
  group_by(subjectid, activity) %>%
  summarise_all(funs(mean))
  
write.table(tidy_average_data, "tidy_average_data.txt", row.names = FALSE, quote = FALSE)
  

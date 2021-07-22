# MARKETING CHANNEL ANALYSIS USING 3 METHODS (Logistic, Random Forest, Decision Tree) ######################################
# A. PROJECT DETAILS ##########################################################
## Dataset: Simple Ads (Multiple Marketing Channel)
## Date   : April 2nd 2021
## Email  : amirudin.adi@gmail.com

# B. PREPARE ##################################################################
# 0.Importing Dataset & Library ###############################################
## 0.1. Importing Dataset #####################################################
datamrkt <- read.csv("D:/1. Active Projects/Data Analysis/Personal Projects Directory/Marketing & Sales Data Predictive Modeling/Dummy Data HSS.csv")

summary(datamrkt)
glimpse(datamrkt)
## 0.2. Importing Package Library #############################################
library(dplyr)
library(tidyr)
library(caret)
library(ggplot2)
library(psych)
library(mice)
library(InformationValue)
library(ROSE)
library(rpart)
library(randomForest)
library(highcharter)
library(prettydoc)


                                      ####


# 1. Data Preparation #########################################################
## 1.1. Formatting Data Types #################################################
## 1.1.1. String Variable to Categorical ######################################
datamrkt$Influencer <- as.factor(datamrkt$Influencer)

## 1.1.2 Formatting Dependent Variable ########################################
Salesbinary <- if_else(datamrkt$Sales >= 272.5, 1, 0)

datamrkt$Salesbinary <- Salesbinary
summary(datamrkt)
#Using Sales 3rd Qu. to Max. as a proxy of good sales and under 3rd Qu as a mediore or bad sales.

summary(datamrkt)
glimpse(datamrkt)

## 1.2.Data Integrity Checking ################################################
sapply(datamrkt, anyNA) #Data contains missing value except for Influencer

## 1.3. Missing Value Treatment ###############################################
### 1.3.1. Iteration (Stochastic Regression) ##################################
imp <- mice(datamrkt, method = "norm.nob", m = 1) #Impute data
Data2 <- complete(imp) #Store data 
sapply(Data2, anyNA)
summary(Data2) 

### 1.3.2. Omitted (na(omit)) #################################################
Data <- na.omit(datamrkt)
sapply(Data, anyNA)

### 1.3.3. Missing Value Treatment (Decision) #################################
sum(is.na(datamrkt))
mean(is.na(datamrkt))
# Since the mean of missing value is relatively low (0.001137358), the na.omit method is preffered
datamrkt <- Data
rm(Data2)
rm(imp) #Clearing unused dataset


## 1.4. Outliers Moderation ###################################################
### 1.4.1. Setting Up Outliers Function #######################################
datamrkt <- na.omit(datamrkt)
sapply(datamrkt, anyNA)


is_outlier <- function(x) {
  return(x < quantile(x, 0.25) - 1.5 * IQR(x) | 
           x > quantile(x, 0.75) + 1.5 * IQR(x)) 
} 

### 1.4.2. Identifying Outliers Count (Create Outlier Dataframe) ##############
outlier <- data.frame(variable = character(), 
                      sum_outliers = integer(),
                      stringsAsFactors=FALSE)
for (j in 1:(length(datamrkt)-1)){
  variable <- colnames(datamrkt[j])
  for (i in datamrkt[j]){
    sum_outliers <- sum(is_outlier(i))
  }
  row <- data.frame(variable,sum_outliers)
  outlier <- rbind(outlier, row)
}

### 1.4.3. Identifying Outliers Percentage ####################################
Treshold.Check <- for (i in 1:nrow(outlier)){
  if (outlier[i,2]/nrow(datamrkt) * 100 >= 5){
    print(paste(outlier[i,1], 
                "The outliers are above 5% treshold", 
                round(outlier[i,2]/nrow(datamrkt) * 100, digits = 2),
                '%'))
  }
}

Treshold.Check
#Since the outliers falls under 5% of the total observation counts, (even under 1%),thus outliers moderation method not necessarily needed.
rm(outlier)
rm(row)
rm(j)
rm(i)
rm(sum_outliers)
rm(variable)
rm(is_outlier)
# cleaning up the environment

                                     ####


# C. ANALYSIS #################################################################
# 2. Preliminary Look at The Data #############################################
## 2.1. Summary Statistics ####################################################
summary(Data)
head(Data)
glimpse(Data)

## 2.2. Univariate Analysis (Dependent Variable) ##############################
### 2.2.1. Frequency Plot
histogram(Data$Sales, Data,
          main = sprintf('Frequency Plot of The Variable: %s', 
                         colnames(Data[5])),
          xlab = colnames(Data[5]),
          ylab = 'Frequency')

### 2.2.2. Check Class BIAS
table(datamrkt$Salesbinary)
round(prop.table((table(datamrkt$Salesbinary))),2)

## 2.3. Univariate Analysis (Independent Variables) ###########################
### 2.3.1. Boxplots of Independent Variables ##################################
par(mfrow=c(3,2))
  for (i in 1:(length(Data)-1)){
    boxplot(x= Data, 
            horizontal = TRUE, 
            main = sprintf('Boxplot of the variable: %s', 
                           colnames(Data[i])),
            xlab = colnames(Data[i]))
  }

## 2.4. Bivariate Analysis (Correlation) ######################################
pairs.panels(Data) #Using pairs plot from psych


                                      ####


# 3. Predictive Modeling ######################################################
## 3.1. Data Splitting (Stratified) ###########################################
data_ones <- Data[which(Data$Sales == 1), ]
data_zeros <- Data[which(Data$Sales == 0), ]

### 3.1.1 Train Data ##########################################################
set.seed(123)
train_ones_rows <- sample(1:nrow(data_ones), 0.8*nrow(data_ones))
train_zeros_rows <- sample(1:nrow(data_zeros), 0.8*nrow(data_ones))
train_ones <- data_ones[train_ones_rows, ]  
train_zeros <- data_zeros[train_zeros_rows, ]
train_set <- rbind(train_ones, train_zeros)

table(train_set$Sales)

### 3.1.2. Test Data ##########################################################
test_ones <- data_ones[-train_ones_rows, ]
test_zeros <- data_zeros[-train_zeros_rows, ]
test_set <- rbind(test_ones, test_zeros)

table(test_set$Sales)

##Test set will be much bigger than train set, however, train set will be balanced to train the model efficiently.

## 3.2. Logistic Regression Prediction ########################################
### 3.2.1. Regression #########################################################
lr = glm(formula = Sales ~.,
         data = train_set,
         family = binomial)
summary(lr)
### 3.2.2. Predictions ########################################################
logi_pred = predict(lr, 
                    type = 'response', 
                    newdata = test_set[-5])
optCutOff <- optimalCutoff(test_set$Sales, logi_pred)[1]
y_pred = ifelse(logi_pred > optCutOff, 1, 0)

## 3.3. Decision Tree Prediction ##############################################
### 3.3.1. Decision Tree ######################################################
dt = rpart(formula = Sales ~ .,
           data = train_set,
           method = 'class')

summary(dt)

### 3.3.2. Prediction #########################################################
dt_pred = predict(dt, 
                 type = 'class', 
                 newdata = test_set[-5])

## 3.4. Random Forest Prediction ##############################################
### 3.4.1. Random Forest ######################################################
rf = randomForest(x = train_set[-5],
                  y = train_set$Sales,
                  ntree = 10)

summary(rf)

### 3.4.2. Prediction #########################################################
rf_pred = predict(rf, 
                 type = 'class', 
                 newdata = test_set[-5])

# 4. Model Evaluation #########################################################
## 4.1. Confusion Matrix ######################################################
### 4.1.1. Logistic Regression ################################################
cm_lr = table(test_set[, 5], y_pred)
cm_lr

### 4.1.2. Decision Tree ######################################################
cm_dt = table(test_set[, 5], dt_pred)
cm_dt

### 4.1.3. Random Forest ######################################################
cm_rf = table(test_set[, 5], rf_pred)
cm_rf

## 4.2. Accuracy ##############################################################
### 4.2.1. Logistic Regression ################################################
accuracy_lr = (cm_lr[1,1] + cm_lr[1,1])/
  (cm_lr[1,1] + cm_lr[1,1] + cm_lr[2,1] + cm_lr[1,2])

accuracy_lr

### 4.2.2. Decision Tree ######################################################
accuracy_dt = (cm_dt[1,1] + cm_dt[1,1])/
  (cm_dt[1,1] + cm_dt[1,1] + cm_dt[2,1] + cm_dt[1,2])

accuracy_dt

### 4.2.3. Random Forest ######################################################
accuracy_rf = (cm_rf[1,1] + cm_rf[1,1])/
  (cm_rf[1,1] + cm_rf[1,1] + cm_rf[2,1] + cm_rf[1,2])

accuracy_rf

## 4.3. ROC Curve #############################################################
### 4.3.1. Logistic Regression ################################################
par(mfrow = c(1, 1))
roc.curve(test_set$Sales, y_pred)

### 4.3.2. Decision Tree ######################################################
roc.curve(test_set$Sales, dt_pred)

### 4.3.3. Random Forest ######################################################
roc.curve(test_set$Sales, rf_pred)

## 4.4. Variable Importance ###################################################
varImp(lr)
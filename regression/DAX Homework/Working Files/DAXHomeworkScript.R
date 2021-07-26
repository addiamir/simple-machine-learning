### DAXHomeworkScriptR ########################################################

### Created by Mohammad Adi Amirudin
### amirudin.adi@gmail.com
### linkedin.com/adi-amirudin

### A. IMPORTING DATASET & LIBRARIES ##########################################
    Data <- read.csv("D:/Data Analysis/DAX Homework/SourceFiles/Data.csv")
    ## Creating backup
    DAXdata <- Data
    attach(DAXdata)
    
    ## Packages Used
    library(tinytex)
    library(psych)
    library(caret)
    library(dplyr)
    library(ggplot2)
    library(GGally)
    library(glmnet)
    library(lars)
    library(elasticnet)
    library(jtools)
    library(knitr)
    library(png)
    
### B. DATA CHECKING ##########################################################
    ## Checking for Missing Values & Data Integrity 
    
    ## Make sure VARIABLES has no missing value: 
    stopifnot(!any(is.na(Math)))
    stopifnot(!any(is.na(Science)))
    stopifnot(!any(is.na(Pysics)))
    stopifnot(!any(is.na(Statistics))) 
    #The results didn't fetch any issue, data has no missing value.
    
    ## Make sure VARIABLES is NUMERIC
    stopifnot(is.numeric(Math))
    stopifnot(is.numeric(Statistics))
    stopifnot(is.numeric(Science))
    stopifnot(is.numeric(Pysics))
    glimpse(DAXdata)
    #data is NUMERIC (no error message fetched), glimpse data describe as integer.

### C. DATA PARTITIONING (CREATING TRAIN DATA) ################################
    ## Creating Training & Test Data
    set.seed(100) 
    index <- sample(1:nrow(DAXdata), 0.7*nrow(DAXdata)) 
    train <- DAXdata[index,] # Create the training data 
    test <- DAXdata[-index,] # Create the test data
    
    ## Check Training & Test Data Dimensions
    dim(train)
    dim(test)
        #Value should be 70% from all observations for training & 30% for test.
        #train = 326 (70% from all DAXdata observation),
        #test = 140 (30% from all DAXdata observation).
    
    ## Determining Independent & Dependent Variable
        # Dependent: Math
        # Independent: Pysics, Science, Statistics

### D. ANALYZING DATA #########################################################
    ## The Data at A Glance (Summary & Description)
    head(DAXdata)
    glimpse(DAXdata)
    summary(DAXdata)
    summary(train)
    summary(test)
    
    ## Data Visualization - Basic
    plot(DAXdata)
    boxplot(DAXdata)
    
    ## Data Visualization - Detailed
    pairs.panels(DAXdata) #taken from package psych
    ggcorr(DAXdata, method = c("pairwise","pearson")) #taken from ggplot
    cor(DAXdata)
        # Visualization, taken from package GGally
          ggpairs(DAXdata[, c(1, 3, 4, 2)],
                  title = "Frequency & Density DAXdata Homework: School Grades",
                  upper = list(continuous = "density", combo = "box_no_facet"),
                  lower = list(continuous = "points", combo = "dot_no_facet"),
                  )
        # Visualization, taken from package ggplot2
          ggpairs(DAXdata[, c(1, 3, 4, 2)], 
                  title="Correlogram DAXdata Homework: School Grades")

### E. MODEL SELECTION ########################################################
    ## Hypothesis 
        #Predict the score of Math using Pysics, Science, Statistics score as predictor. 
        #Ho: Using model "x", the score of Math cannot be predicted.
        #Ha: Using model "x", the score of Math can be predicted.
      
    ## Criteria
        #Model Evaluation: Estimating Performance
        #R** and RMSE of the prediction should be eligible.
        #Prediction result should be comparable with real regression result.
      
    ## Model: Multiple Linear Regression
    linearreg <- lm(Math ~ ., data = DAXdata)
    summary(linearreg)
    lineartrn <- lm(Math ~ ., data = train)
    summary(lineartrn)
        # The results from linearreg & lineartrain statistics, suggests that we can use the model
        # (multiple linear model) for predicting the Math score.
        # Under 95% confidence interval, 2 of 3 predictor significantly increase the Math
        # score, with R** and AdjR** slightly differs from linearreg & lineartrain.
    
    ## Visualization for Model Residual (Base Data)
    ggplot(data=DAXdata, aes(linearreg$residuals)) +
      geom_histogram(binwidth = 1, color = "black", fill = "green") +
      theme(panel.background = element_rect(fill = "#efefef"),
            axis.line.x=element_line(),
            axis.line.y=element_line()) +
      ggtitle("Histogram for Model Residuals Linear")
    
    ## Visualization for Individual Model Residual
    effect_plot(linearreg, pred = Statistics, interval = TRUE, partial.residuals = TRUE,
                main = "Residual Plotting for Math$Statistics")
    effect_plot(linearreg, pred = Science, interval = TRUE, partial.residuals = TRUE,
                main = "Residual Plotting for Math$Science")
    effect_plot(linearreg, pred = Pysics, interval = TRUE, partial.residuals = TRUE,
                main = "Residual Plotting for Math$Pysics")
    
    fit_poly <- lm(Math ~ poly(Statistics, 2) + Pysics + Science, data = DAXdata)
    effect_plot(fit_poly, pred = Statistics, interval = TRUE, plot.points = TRUE)
        # Residuals seems fitted around 0, suggesting our model fits the data well,
        # but we saw some data reach out up to -25 & 25 respectively, and we also
        # have some outlier in -50, we can use some regularization to improve this.

    ## Model accepted as fitted to the data (Ho rejected).

### F. PREPROCESSING INDEPENDENT VARIABLES ####################################
    cols = c('Pysics', 'Science', 'Statistics')
    
    pre_proc_val <- preProcess(train[,cols], method = c("center", "scale"))
    
    train[,cols] = predict(pre_proc_val, train[,cols])
    test[,cols] = predict(pre_proc_val, test[,cols])
    
    summary(train) #Scaling and centering numeric feature
    lineartrain <- lm(Math ~ ., data = train)
    summary(lineartrain)
    
### G. MODEL EVALUATION METRICS ###############################################
    ## Create Model Evaluation Metrics
    eval_metrics <- function(model, df, predictions, target) {
      resids = df[,target] - predictions
      resids2 = resids**2
      N = length(predictions)
      r2 = as.character(round(summary(model)$r.squared, 2))
      adj_r2 = as.character(round(summary(model)$adj.r.squared, 2))
      print("Adjusted R-Squared")
      print(adj_r2) #Adjusted R-squared
      print("RMSE")
      print(as.character(round(sqrt(sum(resids2)/N), 2))) #RMSE
      
    }
    
    ## Compute R** from true and predicted value 
    ## (after regularization for ridge & lasso regression)
    eval_results <- function(true, predicted, df) {
      SSE <- sum((predicted - true)^2)
      SST <- sum((true - mean(true))^2)
      R_square <- 1 - SSE / SST
      RMSE = sqrt(SSE/nrow(df))
      # Model performance metrics
      data.frame(
        RMSE = RMSE,
        Rsquare = R_square
      )
      
      }
    
### H. REGULARIZATION #########################################################
    cols_reg = c('Pysics', 'Science', 'Statistics', 'Math')
    
    dummies <- dummyVars(Math ~., data = DAXdata[,cols_reg])
    
    train_dummies = predict(dummies, newdata = train[,cols_reg])
    
    test_dummies = predict(dummies, newdata = test[,cols_reg])
    
    print(dim(train_dummies)); print(dim(test_dummies))

    # Creating model matrix using dummyVars from glmnet for ridge and lasso,
    # then, predict function applied to the matrices.
    x = as.matrix(train_dummies)
    y_train = train$Math
    
    x_test = as.matrix(test_dummies)
    y_test = test$Math #Creating training data matrices (for ridge & lasso)

### I. REGRESSION MODEL #######################################################
### Manual Regression & Using glmnet package    
    ## Linear Regression
    lineartrain <- lm(Math ~ ., data = train)
    summary(lineartrain)
    
    ggplot(data=train, aes(lineartrain$residuals)) +
      geom_histogram(binwidth = 1, color = "black", fill = "green") +
      theme(panel.background = element_rect(fill = "#efefef"),
            axis.line.x=element_line(),
            axis.line.y=element_line()) +
      ggtitle("Histogram for Model Residuals Linear Training Data")
    
    ## Ridge Regression
    lambdas <- 10^seq(2, -3, by = -.1)
    ridge_reg <- glmnet(x, y_train, nlambda = 25, alpha = 0,
                       #Setting alpha to 0 makes it perform ridge regression
                       family = 'gaussian', lambda = lambdas)
        
    #Ridge regression with optimal lambda
    cv_ridge <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
    optimal_lambda <- cv_ridge$lambda.min
    optimal_lambda
    summary(cv_ridge)
    cv_ridge    
    
    ## Lasso Regression
    lambdas <- 10^seq(2, -3, by = -.1)
    lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, 
                          #Setting alpha to 1 makes it perform lasso regression
                          standardize = TRUE, nfolds = 5)
    lambda_best <- lasso_reg$lambda.min 
    lambda_best
         
    #Lasso Regression using optimal lambda
    lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best, 
                          standardize = TRUE)
    lasso_model
  
### Using caret Package
    ## Linear Regression
    library(caret)
    linearcrt <- train(Math ~., 
                       data = DAXdata,
                       method = "lm")
    linearcrt
         
    ## Hyperparameter tuning (lambda)
    fitControl <- trainControl(method = "repeatedcv",   
                               number = 10,     # number of folds
                               repeats = 10,    # repeated ten times
                               search = "random") #searching for optimal lambda 
        
    ## Ridge Regression
    ridgecrt <- train(Math ~., 
                      data = DAXdata,
                      method = "ridge",
                      trControl = fitControl,
                      preProcess = c('scale','center'),
                      na.action = na.omit)
    ridgecrt
         
    ## Lasso Regression
    lassocrt <- train(Math ~ .,
                      data = DAXdata,
                      method = "lasso",
                      trControl = fitControl,
                      preProcess = c('scale', 'center'),
                      na.action = na.omit)
    lassocrt
         
### J. PREDICTION MODELS ######################################################
### Using manual method & glmnet Package
    #Prediction from linear regression   
    predictlmtrain <- predict(lineartrain, newdata = train)
    predictlmtest <- predict(lineartrain, newdata = test)
         
    #Prediction from ridge regression  
    predictridgetrain <- predict(ridge_reg, s = optimal_lambda, newx = x) 
    predictridgetest <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
         
    #Prediction from lasso regression
    predictlassotrain <- predict(lasso_model, s = lambda_best, newx = x)
    predictlassotest <- predict(lasso_model, s = lambda_best, newx = x_test)
         
### Using caret Package
    #Prediction from linear regression
    predictlmcrt <- predict(linearcrt, DAXdata)
    predictlmcrt
    
    #Prediction from ridge regression
    predictridgecrt <- predict(ridgecrt, DAXdata)
    predictridgecrt      
  
    #Prediction from ridge regression
    predictlassocrt <- predict(lassocrt, DAXdata)
    predictlassocrt

### K. PREDICTION MODEL EVALUATION ############################################
### Using manual regression & glmnet package
    ## Linear Regression
    eval_metrics(lineartrain, train, predictlmtrain, target = 'Math')
    eval_metrics(lineartrain, test, predictlmtest, target = 'Math')
    
    ## Ridge Regression
    eval_results(y_train, predictridgetrain, train)
    eval_results(y_test, predictridgetest, test)
    
    ## Lasso Regression
    eval_results(y_train, predictlassotrain, train)
    eval_results(y_test, predictlassotest, test)
    
### Using caret package
    ## Linear regression
    predictlmcrt
    
    ## Ridge Regression
    predictridgecrt
    
    ## Lasso Regression
    predictlassocrt

### L. CONCLUSIONS ############################################################
    
## Clear packages
    detach("package:datasets", unload = TRUE)  # For base
    
## Clear plots
    dev.off()  # But only if there IS a plot
    
## Clear console
    cat("\014")  # ctrl+L

# Clear mind :)
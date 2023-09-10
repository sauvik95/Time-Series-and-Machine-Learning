library(ISLR2)
library(caret)
library(rsample)
data <- Boston[,-4]
data

## (a) Using 5-fold Cross-Validation to tune our hyperparameter k on entire model
k_values <- expand.grid(k = c(1:50))
cv <- trainControl(method = "cv", number = 5)
knn_cv_fit <- train(medv ~ crim+zn+indus+nox+rm+age+dis+rad+tax+ptratio+lstat,
             data = data,
             trControl = cv,
             tuneGrid = k_values,
             method = "knn")
knn_cv_fit
optimal_k <- knn_cv_fit$bestTune$k
optimal_k
MSE_model <- (knn_cv_fit$results$RMSE[optimal_k])^2
MSE_model

## (b) Estimating test error using Holdout method
k_values <- expand.grid(k=c(1:50))
split <- initial_split(data,prop = 0.8)
train <- training(split)
test <- testing(split)
holdout_method <- trainControl(method = "LGOCV", p = 0.8, number = 1)
fit <- train(medv ~ crim+zn+indus+nox+rm+age+dis+rad+tax+ptratio+lstat,
             data = train,
             trControl = holdout_method,
             tuneGrid = k_values,
             method = "knn")
pred <- predict(fit,newdata = test)
test_error <- mean((test$medv - pred)^2)
test_error

## (c) Predicting medv for new data
new_data <- data.frame(crim = 0.257,zn = 0,indus = 9.69,chas = 0,
                       nox = 0.538,rm = 6.21,age = 77.5,dis = 3.21,
                       rad = 5,tax = 330,ptratio = 19.0,lstat = 11.4)
new_data
new_pred <- predict(fit,newdata = new_data)
new_pred

## (d) Changing lstat from 5 to 10 and analyzing change in medv
new_data_lstat_five <- data.frame(crim = 0.257,zn = 0,indus = 9.69,chas = 0,
                       nox = 0.538,rm = 6.21,age = 77.5,dis = 3.21,
                       rad = 5,tax = 330,ptratio = 19.0,lstat = 5)
new_data_lstat_ten <- data.frame(crim = 0.257,zn = 0,indus = 9.69,chas = 0,
                                  nox = 0.538,rm = 6.21,age = 77.5,dis = 3.21,
                                  rad = 5,tax = 330,ptratio = 19.0,lstat = 10)

new_pred_lstat_five <- predict(fit,newdata = new_data_lstat_five)
new_pred_lstat_five

new_pred_lstat_ten <- predict(fit,newdata = new_data_lstat_ten)
new_pred_lstat_ten

## (e) Change in medv for 5 unit change in lstat

medv_results <- data.frame(lstat = numeric(0), pred_medv = numeric(0))

# Loop through lstat values from 5 to 25 with increments of 5
for (lstat_value in seq(5, 25, by = 5)) {
  new_lstat_data <- data.frame(crim = 0.257,zn = 0,indus = 9.69,
                               chas = 0,nox = 0.538,rm = 6.21,
                               age = 77.5,dis = 3.21,rad = 5,
                               tax = 330,ptratio = 19.0,lstat = lstat_value)
  
  new_medv_pred <- predict(fit, newdata = new_lstat_data)
  medv_results <- rbind(medv_results, data.frame(lstat = lstat_value, pred_medv = new_medv_pred))
}

# Final results
medv_results

# No, we do not see same amount of change in medv as lstat changes
# medv prediction decreases as lstat value increases by 5 units
# This signifies that when lower status (lstat) population increases in an area, our median-value of owner occupied home (medv) decreases as less people would want to stay in that area


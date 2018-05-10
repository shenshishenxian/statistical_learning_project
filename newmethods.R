# This code contains all the new approaches we tried

# Read Data
library(readr)
diabetes <- read_csv("/home/yuxin/Documents/StatLearning/Project/data/diabetes.csv")
diabetes.factor <- diabetes
diabetes.factor$Outcome <- as.factor(diabetes$Outcome)

# Training Testing Split
trainIndex <- createDataPartition(diabetes$Outcome, p=0.80, list=FALSE)
data_train <- diabetes[ trainIndex,]
data_test <- diabetes[-trainIndex,]

# 1. No Preprocessing
## 1.1 Naive Bayse
library(e1071)
NBclassfier <- naiveBayes(Outcome~., data=diabetes.factor[ trainIndex,])
predictions <- predict(NBclassfier, diabetes.factor[-trainIndex,])
confusionMatrix(predictions, diabetes.factor[-trainIndex,]$Outcome)
## 1.2 K-NN
library(class)
kNN_prediction <- knn(data_train, data_test, data_train$Outcome, k = 22)
confusionMatrix(as.factor(kNN_prediction), as.factor(data_test$Outcome))
## 1.3 Neural Network
library(neuralnet)
nn <- neuralnet::neuralnet(Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + 
                  BMI + DiabetesPedigreeFunction + Age, data_train, hidden=7,
                algorithm = "rprop+", err.fct = "sse", act.fct = "logistic",
                threshold = 0.03, linear.output=FALSE, lifesign = "full")
temp_test <- subset(data_test, select = c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                                          "BMI", "DiabetesPedigreeFunction", "Age"))
predictions <- neuralnet::compute(nn, temp_test)$net.result
predictions[predictions > 0.5,] = 1
predictions[predictions <= 0.5] = 0
confusionMatrix(as.factor(predictions), as.factor(data_test$Outcome))
## 1.4 XGBoost
library(xgboost)
temp_train_data <- subset(data_train, select = c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                                          "BMI", "DiabetesPedigreeFunction", "Age"))
temp_train_label <- t(subset(data_train, select = c("Outcome")))
temp_test_data <- subset(data_test, select = c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                                                "BMI", "DiabetesPedigreeFunction", "Age"))
temp_test_label <- t(subset(data_test, select = c("Outcome")))
boost <- xgboost(data = as.matrix(temp_train_data), label = temp_train_label, 
                 max.depth = 2, eta = 1, nthread = 2, nround = 20, objective = "binary:logistic")
predictions <- predict(boost, as.matrix(temp_test_data))
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
confusionMatrix(as.factor(predictions), as.factor(data_test$Outcome))

# 2. Normalization
train_data.scale <- scale(temp_train_data)
train_label <- temp_train_label
test_data.scale <- scale(temp_test_data)
test_label <- temp_test_label
diabetes.train.scale <- data.frame(train_data.scale, train_label)
diabetes.test.scale <- data.frame(test_data.scale, test_label)
diabetes.train.scale.factor <- data.frame(train_data.scale, as.factor(train_label))
diabetes.test.scale.factor <- data.frame(test_data.scale, as.factor(test_label))
colnames(diabetes.train.scale.factor) <- colnames(diabetes)
colnames(diabetes.test.scale.factor) <- colnames(diabetes)
## 2.1 Naive Bayse
NBclassfier <- naiveBayes(Outcome~., data=diabetes.train.scale.factor)
predictions <- predict(NBclassfier, diabetes.test.scale.factor)
confusionMatrix(predictions, diabetes.test.scale.factor$Outcome)
## 2.2 K-NN
kNN_prediction <- knn(diabetes.train.scale.factor, diabetes.test.scale.factor, diabetes.train.scale.factor$Outcome, k = 22)
confusionMatrix(as.factor(kNN_prediction), as.factor(diabetes.test.scale.factor$Outcome))
## 2.3 Neural Network
nn <- neuralnet::neuralnet(Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + 
                             BMI + DiabetesPedigreeFunction + Age, data_train, hidden=9,
                           algorithm = "rprop+", err.fct = "sse", act.fct = "logistic",
                           threshold = 0.03, linear.output=FALSE, lifesign = "full")
predictions <- neuralnet::compute(nn, test_data.scale)$net.result
predictions[predictions > 0.5,] = 1
predictions[predictions <= 0.5] = 0
confusionMatrix(as.factor(predictions), as.factor(test_label))
## 2.4 XGBoost
boost <- xgboost(data = as.matrix(train_data.scale), label = train_label, 
                 max.depth = 2, eta = 1, nthread = 2, nround = 20, objective = "binary:logistic")
predictions <- predict(boost, as.matrix(test_data.scale))
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
confusionMatrix(as.factor(predictions), as.factor(test_label))

# 3. Discretization
library(arules)
diabetes.train.discrete <- data_train
diabetes.train.discrete$Pregnancies <- as.numeric(discretize(data_train$Pregnancies, method = "fixed", breaks = c(-Inf, 0, 1, 5, Inf), 
                                                               labels = c(0, 1, 2, 3)))
diabetes.train.discrete$Glucose <- as.numeric(discretize(data_train$Glucose, method = "fixed", breaks = c(-Inf, 0, 94, 140, Inf), 
                                                           labels = c(0, 1, 2, 3)))
diabetes.train.discrete$BloodPressure <- as.numeric(discretize(data_train$BloodPressure, method = "fixed", breaks = c(-Inf, 0, 80, 89, Inf), 
                                                                 labels = c(0, 1, 2, 3)))
diabetes.train.discrete$SkinThickness <- data_train$SkinThickness
diabetes.train.discrete$Insulin <- data_train$Insulin
diabetes.train.discrete$BMI <- as.numeric(discretize(data_train$BMI, method = "fixed", breaks = c(-Inf, 0, 18.5, 24.9, 29.9, 34.9, Inf), 
                                                       labels = c(0, 1, 2, 3, 4, 5)))
diabetes.train.discrete$DiabetesPedigreeFunction <- as.numeric(discretize(data_train$DiabetesPedigreeFunction, method = "fixed", 
                                                                            breaks = c(-Inf, 0, 0.41, 0.82, Inf), labels = c(0, 1, 2, 3)))
diabetes.train.discrete$Age <- as.numeric(discretize(data_train$Age, method = "fixed", breaks = c(-Inf, 0, 40, 61, Inf), 
                                                       labels = c(0, 1, 2, 3)))
diabetes.train.discrete$Outcome <- data_train$Outcome

diabetes.test.discrete <- data_test
diabetes.test.discrete$Pregnancies <- as.numeric(discretize(data_test$Pregnancies, method = "fixed", breaks = c(-Inf, 0, 1, 5, Inf), 
                                                             labels = c(0, 1, 2, 3)))
diabetes.test.discrete$Glucose <- as.numeric(discretize(data_test$Glucose, method = "fixed", breaks = c(-Inf, 0, 94, 140, Inf), 
                                                         labels = c(0, 1, 2, 3)))
diabetes.test.discrete$BloodPressure <- as.numeric(discretize(data_test$BloodPressure, method = "fixed", breaks = c(-Inf, 0, 80, 89, Inf), 
                                                               labels = c(0, 1, 2, 3)))
diabetes.test.discrete$SkinThickness <- data_test$SkinThickness
diabetes.test.discrete$Insulin <- data_test$Insulin
diabetes.test.discrete$BMI <- as.numeric(discretize(data_test$BMI, method = "fixed", breaks = c(-Inf, 0, 18.5, 24.9, 29.9, 34.9, Inf), 
                                                     labels = c(0, 1, 2, 3, 4, 5)))
diabetes.test.discrete$DiabetesPedigreeFunction <- as.numeric(discretize(data_test$DiabetesPedigreeFunction, method = "fixed", 
                                                                          breaks = c(-Inf, 0, 0.41, 0.82, Inf), labels = c(0, 1, 2, 3)))
diabetes.test.discrete$Age <- as.numeric(discretize(data_test$Age, method = "fixed", breaks = c(-Inf, 0, 40, 61, Inf), 
                                                     labels = c(0, 1, 2, 3)))
diabetes.test.discrete$Outcome <- data_test$Outcome

diabetes.train.discrete.factor <- diabetes.train.discrete
diabetes.train.discrete.factor$Outcome <- as.factor(diabetes.train.discrete$Outcome)
diabetes.test.discrete.factor <- diabetes.test.discrete
diabetes.test.discrete.factor$Outcome <- as.factor(diabetes.test.discrete$Outcome)
colnames(diabetes.train.discrete.factor) <- colnames(diabetes)
colnames(diabetes.test.discrete.factor) <- colnames(diabetes)
## 3.1 Naive Bayse
NBclassfier <- naiveBayes(Outcome~., data=diabetes.train.discrete.factor)
predictions <- predict(NBclassfier, diabetes.test.discrete.factor)
confusionMatrix(predictions, diabetes.test.discrete.factor$Outcome)
## 3.2 K-NN
kNN_prediction <- knn(diabetes.train.discrete.factor, diabetes.test.discrete.factor, diabetes.train.discrete.factor$Outcome, k = 22)
confusionMatrix(as.factor(kNN_prediction), as.factor(diabetes.test.discrete.factor$Outcome))
## 3.3 Neural Network
nn <- neuralnet::neuralnet(Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + 
                             BMI + DiabetesPedigreeFunction + Age, diabetes.train.discrete, hidden=7,
                           algorithm = "rprop+", err.fct = "sse", act.fct = "logistic",
                           threshold = 0.03, linear.output=FALSE, lifesign = "full")
temp_test <- subset(diabetes.test.discrete, select = c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                                          "BMI", "DiabetesPedigreeFunction", "Age"))
predictions <- neuralnet::compute(nn, temp_test)$net.result
predictions[predictions > 0.5,] = 1
predictions[predictions <= 0.5] = 0
confusionMatrix(as.factor(predictions), as.factor(test_label))
## 3.4 XGBoost
temp_train_data <- subset(diabetes.train.discrete, select = c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                                                 "BMI", "DiabetesPedigreeFunction", "Age"))
temp_train_label <- t(subset(diabetes.train.discrete, select = c("Outcome")))
temp_test_data <- subset(diabetes.test.discrete, select = c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                                               "BMI", "DiabetesPedigreeFunction", "Age"))
temp_test_label <- t(subset(diabetes.test.discrete, select = c("Outcome")))
boost <- xgboost(data = as.matrix(temp_train_data), label = temp_train_label, 
                 max.depth = 2, eta = 1, nthread = 2, nround = 20, objective = "binary:logistic")
predictions <- predict(boost, as.matrix(temp_test_data))
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
confusionMatrix(as.factor(predictions), as.factor(temp_test_label))

# 4. Discretization + Normalization
diabetes.train.discrete.data <- subset(diabetes.train.discrete, select = c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                                                                           "BMI", "DiabetesPedigreeFunction", "Age"))
diabetes.train.discrete.label <-  subset(diabetes.train.discrete, select = c("Outcome"))
diabetes.test.discrete.data <- subset(diabetes.test.discrete, select = c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                                                            "BMI", "DiabetesPedigreeFunction", "Age"))
diabetes.test.discrete.label <- subset(diabetes.test.discrete, select = c("Outcome"))

diabetes.train.discrete.data.scale <- scale(diabetes.train.discrete.data)
diabetes.test.discrete.data.scale <- scale(diabetes.test.discrete.data)

diabetes.train.discrete.scale <- data.frame(diabetes.train.discrete.data.scale, diabetes.train.discrete.label)
diabetes.test.discrete.scale <- data.frame(diabetes.test.discrete.data.scale, diabetes.test.discrete.label)

diabetes.train.discrete.scale.factor <- data.frame(diabetes.train.discrete.data.scale, as.factor(diabetes.train.discrete.label))
diabetes.test.discrete.scale.factor <- data.frame(diabetes.test.discrete.data.scale, as.factor(diabetes.test.discrete.label))
colnames(diabetes.train.discrete.scale.factor) <- colnames(diabetes)
colnames(diabetes.test.discrete.scale.factor) <- colnames(diabetes)
## 4.1 Naive Bayse
NBclassfier <- naiveBayes(Outcome~., data=diabetes.train.discrete.scale.factor)
predictions <- predict(NBclassfier, diabetes.test.discrete.scale.factor)
confusionMatrix(predictions, diabetes.test.discrete.scale.factor$Outcome)
## 4.2 K-NN
kNN_prediction <- knn(diabetes.train.discrete.scale.factor, diabetes.test.discrete.scale.factor, diabetes.train.discrete.scale.factor$Outcome, k = 22)
confusionMatrix(as.factor(kNN_prediction), as.factor(diabetes.test.discrete.scale.factor$Outcome))
## 4.3 Neural Network
nn <- neuralnet::neuralnet(Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + 
                             BMI + DiabetesPedigreeFunction + Age, diabetes.train.discrete.scale, hidden=7,
                           algorithm = "rprop+", err.fct = "sse", act.fct = "logistic",
                           threshold = 0.03, linear.output=FALSE, lifesign = "full")
predictions <- neuralnet::compute(nn, diabetes.test.discrete.data.scale)$net.result
predictions[predictions > 0.5,] = 1
predictions[predictions <= 0.5] = 0
confusionMatrix(as.factor(predictions), as.factor(diabetes.test.discrete.scale$Outcome))
## 4.4 XGBoost
boost <- xgboost(data = as.matrix(diabetes.train.discrete.data), label = t(diabetes.train.discrete.label), 
                 max.depth = 2, eta = 1, nthread = 2, nround = 20, objective = "binary:logistic")
predictions <- predict(boost, as.matrix(diabetes.test.discrete.data))
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
confusionMatrix(as.factor(predictions), as.factor(diabetes.test.discrete.scale$Outcome))
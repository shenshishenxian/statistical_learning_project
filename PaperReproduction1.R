# This code reproduces the results from the first paper
# Diabetes Data Analysis And Prediction Model Discovery Using Rapidminer

# Read Data
library(readr)
diabetes <- read_csv("/home/yuxin/Documents/StatLearning/Project/data/diabetes.csv")

# Data Preprocessing
## Outlier removal and feature selection 
diabetes.process <- diabetes[diabetes$Glucose != 0 & diabetes$BloodPressure != 0& diabetes$BMI!= 0,]

## Data Discretization
library(arules)
diabetes.process.discrete <- diabetes.process
diabetes.process.discrete$Pregnancies <- as.numeric(discretize(diabetes.process$Pregnancies, method = "fixed", breaks = c(-Inf, 0, 1, 5, Inf), 
                                           labels = c(0, 1, 2, 3)))
diabetes.process.discrete$Glucose <- as.numeric(discretize(diabetes.process$Glucose, method = "fixed", breaks = c(-Inf, 0, 94, 140, Inf), 
                                       labels = c(0, 1, 2, 3)))
diabetes.process.discrete$BloodPressure <- as.numeric(discretize(diabetes.process$BloodPressure, method = "fixed", breaks = c(-Inf, 0, 80, 89, Inf), 
                                             labels = c(0, 1, 2, 3)))
diabetes.process.discrete$SkinThickness <- diabetes.process$SkinThickness
diabetes.process.discrete$Insulin <- diabetes.process$Insulin
diabetes.process.discrete$BMI <- as.numeric(discretize(diabetes.process$BMI, method = "fixed", breaks = c(-Inf, 0, 18.5, 24.9, 29.9, 34.9, Inf), 
                                   labels = c(0, 1, 2, 3, 4, 5)))
diabetes.process.discrete$DiabetesPedigreeFunction <- as.numeric(discretize(diabetes.process$DiabetesPedigreeFunction, method = "fixed", 
                                                                 breaks = c(-Inf, 0, 0.41, 0.82, Inf), labels = c(0, 1, 2, 3)))
diabetes.process.discrete$Age <- as.numeric(discretize(diabetes.process$Age, method = "fixed", breaks = c(-Inf, 0, 40, 61, Inf), 
                                            labels = c(0, 1, 2, 3)))
diabetes.process.discrete$Outcome <- diabetes.process$Outcome

## Data Normalization
diabetes.scaled <- diabetes.process.discrete
diabetes.scaled$Pregnancies <- as.factor(scale(diabetes.process.discrete$Pregnancies))
diabetes.scaled$Glucose <- as.factor(scale(diabetes.process.discrete$Glucose))
diabetes.scaled$BloodPressure <- as.factor(scale(diabetes.process.discrete$BloodPressure))
diabetes.scaled$SkinThickness <- as.factor(scale(diabetes.process.discrete$SkinThickness))
diabetes.scaled$Insulin <- as.factor(scale(diabetes.process.discrete$Insulin))
diabetes.scaled$BMI <- as.factor(scale(diabetes.process.discrete$BMI))
diabetes.scaled$DiabetesPedigreeFunction <- as.factor(scale(diabetes.process.discrete$DiabetesPedigreeFunction))
diabetes.scaled$Age <- as.factor(scale(diabetes.process.discrete$Age))
diabetes.scaled$Outcome <- as.factor(diabetes.process.discrete$Outcome)

# Loading ID3 algorithm
library(RWeka)
WPM("refresh-cache") 
WPM("list-packages", "available") 
WPM("install-package", "simpleEducationalLearningSchemes") 
WPM("load-package", "simpleEducationalLearningSchemes") 
ID3 <- make_Weka_classifier("weka/classifiers/trees/Id3") 

# Train ID3 Algorithm
id3.diabetes <- ID3(Outcome ~. ,data = diabetes.scaled[1:500,])
summary(id3.diabetes)

# Testing Data and Testing Accuracy
pred1 <- predict(id3.diabetes, diabetes.scaled[501:724,], type = "class")
table(pred1, diabetes.scaled[501:724,]$Outcome)

# Data Accuracy
pred2 <- predict(id3.diabetes, diabetes.scaled, type = "class")
table(pred2, diabetes.scaled$Outcome)

# Read Data
library(readr)
diabetes <- read_csv("diabetes.csv")

# Data Preprocessing
## Outlier removal and feature selection 
diabetes.process <- diabetes[diabetes$Glucose != 0 & diabetes$BloodPressure != 0& diabetes$BMI!= 0&diabetes$Insulin != 0
                             $SkinThickness != 0& diabetes$DiabetesPedigreeFunction!= 0,]

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
#using Weka interface doing k-means and saving the k-means prediction result
library(RWeka)
cl1 <- SimpleKMeans(diabetes.process.discrete[, ], Weka_control(N = 2))
write.csv(predict(cl1), file = "MyData.csv")
#converting the data to right form which can be processed in the following step, diabetedelete is the samples have the same label with MyData.csv
mydata = read.csv("diabetesdelete.csv", header = 0)
names(mydata)<-c("pregnancy","plasma","bloodpressure","skinthickness","insulin","bodymass","diabete","age" ,"class")
mydata$class[mydata$class==0]<-'A'
mydata$class[mydata$class==1]<-'B'
mydata$plasma<-factor(mydata$plasma)
mydata$insulin<-factor(mydata$insulin)
mydata$skinthickness<-factor(mydata$skinthickness)
mydata$age<-factor(mydata$age)
mydata$diabete<-factor(mydata$diabete)
mydata$bodymass<-factor(mydata$bodymass)
mydata$pregnancy<-factor(mydata$pregnancy)
mydata$bloodpressure<-factor(mydata$bloodpressure)
mydata$class<-factor(mydata$class)

#constructing ctree and did the preprocessing
ctree<-J48(class~.,data=mydata[0:180,],control=Weka_control(M=2))
print(ctree)
library(partykit)
plot(ctree,type="simple")
pretree<-predict(ctree,mydata[180:300,])
table(pretree,mydata[180:300,]$class,dnn=c("Prediction","True Value"))
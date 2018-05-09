---
title: "R Notebook"
output: html_notebook
---

This is an [R Markdown](http://rmarkdown.rstudio.com) Notebook. When you execute code within the notebook, the results appear beneath the code. 

Try executing this chunk by clicking the *Run* button within the chunk or by placing your cursor inside it and pressing *Cmd+Shift+Enter*. 
```{r 1}
#importing the data and using Weka interface doing k-means
library(RWeka)
mydata = read.csv("afterprocessfirst.csv", header = 0)
cl1 <- SimpleKMeans(mydata[, ], Weka_control(N = 2))
write.csv(predict(cl1), file = "MyData.csv")
```


```{r}
#converting the data to right form which can be processed in the following step
library(RWeka)
mydata = read.csv("afterprocess.csv", header = 0)
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
# 
# train<-sample(nrow(mydata),0.6*nrow(mydata))
# #tdata<-data[train,]
# #vdata<-data[-train,]  
# a<-round(3/4*sum(mtcars$group=='A'))
# b<-round(3/4*sum(mtcars$group=='B'))
# library(sampling) 
# sub<-strata(mydata,stratanames="class",size=c(a,b),method="srswor")
# tdata<-mydata[sub$ID,]   #训练集
# vdata<-mydata[-sub$ID,]   #预测集

#constructing ctree and did the preprocessing
ctree<-J48(class~.,data=mydata[0:180,],control=Weka_control(M=2))
print(ctree)
library(partykit)
plot(ctree,type="simple")
pretree<-predict(ctree,mydata[180:300,])
table(pretree,mydata[180:300,]$class,dnn=c("Prediction","True Value"))
```

Add a new chunk by clicking the *Insert Chunk* button on the toolbar or by pressing *Cmd+Option+I*.

When you save the notebook, an HTML file containing the code and output will be saved alongside it (click the *Preview* button or press *Cmd+Shift+K* to preview the HTML file). 

The preview shows you a rendered HTML copy of the contents of the editor. Consequently, unlike *Knit*, *Preview* does not run any R code chunks. Instead, the output of the chunk when it was last run in the editor is displayed.


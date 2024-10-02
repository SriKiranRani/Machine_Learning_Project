rm(list = ls())
cat("\014")
setwd("C:/Users/ravip/Desktop/SEM1/business analy with R/project")

Data_all = read.csv("framingham.csv")
colnames(Data_all)[colnames(Data_all)=="male"]= "Sex"
write.csv(Data_all,file = "framingham.csv",row.names =F )
df = Data_all

head(df)
colnames(df)
str(df)
summary(is.na(df))

#changing all the categorical  variables to factors
df$Sex = as.factor(df$Sex)
df$education = as.factor(df$education)
df$currentSmoker = as.factor(df$currentSmoker)
df$BPMeds = as.factor(df$BPMeds)
df$prevalentStroke = as.factor(df$prevalentStroke)
df$prevalentHyp = as.factor(df$prevalentHyp)
df$diabetes= as.factor(df$diabetes)

#labeling my target variable binary data to characters 
df[df$TenYearCHD != 0,"TenYearCHD"] = "Risk_of_CHD"
df[df$TenYearCHD == 0,"TenYearCHD"] = "No_Risk"


df$TenYearCHD = as.factor(df$TenYearCHD)

str(df)

#-------------------------


#eliminating all the missing vlues in my dataset
df.nona <- na.omit(df) #dataset without missing values

Data1 = df[complete.cases(df),] #you can use this line too to eliminate na values
# 
#observing how each variable relates with target variable(TenYearCHD) 
xtabs(~Sex+TenYearCHD,data= Data1)
xtabs(~currentSmoker+TenYearCHD,data= Data1)
xtabs(~BPMeds+TenYearCHD,data= Data1)
xtabs(~prevalentStroke+TenYearCHD,data= Data1)
xtabs(~prevalentHyp+TenYearCHD,data= Data1)
xtabs(~diabetes+TenYearCHD,data= Data1)


library(caret)
# create binary output Y (consumer rating) for later use
Data1$TenYearCHD <- as.numeric(Data1$TenYearCHD == "Risk_of_CHD")

Data1$TenYearCHD = as.factor(Data1$TenYearCHD)

# specify which columns to one-hot encode
cat_cols <- c("Sex","education","currentSmoker","BPMeds", "prevalentStroke", "prevalentHyp", "diabetes")

# create dummy variables
dummy_obj <- dummyVars(~., data=Data1[,cat_cols], fullRank=FALSE)

# apply the transformation to the original dataset
Data1_encoded <- predict(dummy_obj, newdata=Data1)

# merge the encoded variables with the original dataset
Data1 <- cbind(Data1, Data1_encoded)



# split the data into training and test data sets
set.seed(123)   # for reproducible results
train <- sample(1:nrow(Data1), (0.7)*nrow(Data1))
train.df <- Data1[train,-c(1,3,4,6,7,8,9)] # keep all columns but "rating"
test.df <- Data1[-train, -c(1,3,4,6,7,8,9)] # keep all columns but "rating"



################  Principal Component Analysis################  
### perform PCA for all numerical variables in the training data ###
pca_out_1 <- prcomp(na.omit(train.df[,-c(9)], scale. = T))  # leave out non-numerical variables and Y
# summary of PCs
summary(pca_out_1) 

# let's look at the weights of these components, stored in "rotation" of the output
# map from each PC to the original variables
pca_out_1$rotation

# get principal component scores from "x" of the output 
scores <- pca_out_1$x
head(scores, 5)
# correlations are zero between different pcs, as each PC is diagonal to another
cor(scores[, 1:3])  # the first 3 PCs




### lets visualize ###
library(ggplot2)

# scree plot: plot of the proportion of variance explained (PVE) by each PC
# first let us create vector of variances for each generated principal component
pca.var <- pca_out_1$sdev^2
# this is what we had in the summary: i.e., proportion of variance explained by each principal component
pca.pve <- data.frame( pve = pca.var/sum(pca.var), component = c(1:24))  
# plot
plot(pca.pve$pve)  # the "elbow" of the graph <=> the number of PCs to keep
g <- ggplot(pca.pve, aes(component, pve))  
g + geom_point() + labs(title="Scree Plot", x="Component Number", y="PVE") +
  scale_x_continuous(breaks=seq(1,24,by=1))

# plot the weight of the original variables in each PC
rot <- as.data.frame(pca_out_1$rotation)  # loadings: rows are variable, columns are components
# add column "variable" which records the rownames
rot$feature <- rownames(rot)

# ordered in decreasing loading (i.e., weight) for PC1
rot$feature <- factor(rot$feature, levels= rot$feature[order(rot$PC1, decreasing=T)])
ggplot(rot, aes(feature, PC1)) + 
  geom_bar(stat="identity", position="identity") +   #basic bar plot
  theme(axis.text.x=element_text(size = 15, angle=45, hjust=1)) # fix x axis labels

#The values of principal component (PC) scores can be both positive and negative. 
#In fact, it is common for principal component scores to have both positive and negative values. 
#The sign of the score only indicates the direction of the variation that the principal component represents.

#In your case, the PC1 values have both positive and negative values, with some values being close to zero.
# The magnitude of the value (i.e., how far it is from zero) indicates the strength of the variation in that direction. 
# The interpretation of principal components involves examining which variables are most strongly correlated with each 
# principal component to understand what kind of variation is being captured.


# ordered in decreasing loading for PC2
rot$feature <- factor(rot$feature, levels= rot$feature[order(rot$PC2, decreasing=T)])
ggplot(rot, aes(feature, PC2)) + 
  geom_bar(stat="identity", position="identity") +   #basic bar plot
  theme(axis.text.x=element_text(size=15, angle=45, hjust=1)) # fix x axis labels

################  PCA + Classification ################  
# append pca scores to train.df 
train.df <- cbind(train.df, pca_out_1$x)
# calculate pca scores for the testing data using predict()  
testScores <- as.data.frame(predict(pca_out_1,test.df))
test.df <- cbind(test.df, testScores)
rm(testScores)


# use glm() (general linear model) with family = "binomial" to fit a logistic 
logit.reg <- glm(TenYearCHD ~ PC1+PC2+PC3+PC4+PC5,
                 data = train.df, family = "binomial") 
summary(logit.reg)

# use predict() with type = "response" to compute predicted probabilities. 
logitPredict <- predict(logit.reg, test.df, type = "response")
# we choose 0.5 as the cutoff here for 1 vs. 0 classes
logitPredictClass <- ifelse(logitPredict > 0.5, 1, 0)

# Evaluate classifier performance on testing data
library(caret)
actual <- test.df$TenYearCHD
predict <- logitPredictClass
confusionMatrix(factor(predict), actual,positive= "1")
#Accuracy : 0.913 Balanced Accuracy : 0.9231           



#-----------------------------------------
#       LOGISTIC REGRESSION
#-----------------------------------------       
#checking logistic regression deviation performance using only sex variable
logistic= glm(TenYearCHD~Sex, data= df.nona, family = "binomial" )
summary(logistic)

#performing logistic regression taking all the variables at a time
logistic= glm(TenYearCHD~., data= df.nona, family = "binomial" )
summary(logistic)


# Calculate log likelihoods of null deviance and proposed deviance from logistic regression
ll.null = logistic$null.deviance/(-2);ll.null
ll.proposed = logistic$deviance/(-2);ll.proposed

# Calculate R squared value from logistic regression
R.sq = (ll.null-ll.proposed)/(ll.null);R.sq

# Calculate p-value for logistic regression
p.val = pchisq(2*(ll.proposed-ll.null),df = (length(logistic$coefficients)-1),lower.tail = F);p.val


# Create a data frame of predicted probabilities and observed CHD
predicted_data = data.frame(probability_of_CHD =logistic$fitted.values,CHD = Data1$TenYearCHD);predicted_data

predicted_data$predicted_CHD = ifelse(predicted_data$probability_of_CHD>=0.5,"Risk_of_CHD","No_Risk");predicted_data


accuracy= round(sum(predicted_data$predicted_CHD==predicted_data$CHD)/nrow(predicted_data),5);accuracy
LR_confusionMatrix = table (predicted_data$predicted_CHD, predicted_data$CHD); LR_confusionMatrix


a = LR_Prop_tab= prop.table(LR_confusionMatrix)
LR_Prop_tab

# LRaccuracy
LR_accuracy = LR_Prop_tab[1,1] + LR_Prop_tab[2,2]   # 0.8558534


FPR = a[2,1];FPR

FNR = a[1,2];FNR

# Calculate sensitivity (true positive rate)
sensitivity <- LR_confusionMatrix[2, 2] / sum(LR_confusionMatrix[, 2]);sensitivity #sensitivity/ recall 0.08797

# Calculate specificity (true negative rate)
specificity <- LR_confusionMatrix[1, 1] / sum(LR_confusionMatrix[, 1]);specificity #sensitivity 0.9938

# Calculate precision (positive predictive value)
precision <- LR_confusionMatrix[2, 2] / sum(LR_confusionMatrix[2, ]);precision #precision 0.72058
# Low recall value with high precision value suggests that, model successfully predicted the most true positives.
# in the same tie it failed to capture the actual positives



#To develop a graphical representation with type 1
predicted_data1 = predicted_data[order(predicted_data$probability_of_CHD, decreasing = F),]

predicted_data1$rank = 1:nrow(predicted_data1)

library(ggplot2)
ggplot(data = predicted_data1,aes(x=rank,y=probability_of_CHD))+
  geom_point(aes(color= CHD),alpha =1, shape = 4,stroke = 2)+
  xlab("Index")+
  ylab("predicted probability of having a CHD")

#second type 2
predicted_data$rank = 1:nrow(predicted_data)

ggplot(data = predicted_data,aes(x=rank,y=probability_of_CHD))+
  geom_point(aes(color= CHD),alpha =1, shape = 4,stroke = 2)+
  xlab("Index")+
  ylab("predicted probability of having a CHD")

#----------------------------------------
#         DECISION TREE
#----------------------------------------
library(caret)

#---------------------------------------------------
Data_DT = read.csv("framingham.csv")
Data_DT = Data_DT[complete.cases(df),]
#cross validation
set.seed(136)


train= sample(1:nrow(Data_DT),(2/3)*(nrow(Data_DT)))

train_data = Data_DT[train,]
test_data = Data_DT[-train,]

library(rpart)

fit= rpart(TenYearCHD~.,
           data = train_data,
           method = "class",
           control= rpart.control(xval= 10,minsplit= 20,cp=0.005),
           parms = list(split = "gini") )
fit

best_cp = printcp(fit)

summary(Data1)

#class imbalance- majority of the predicted model is dominated by no_risk class.
#but for us it is more imp to determine the risk rather than no_risk, so we have to balance the dataset

barplot(prop.table(table(Data1$TenYearCHD)),
        col = rainbow(2),
        ylim= c(0,0.9),
        main= "Class Distribution")


#------------------------------------------
#         Random Forest
#------------------------------------------

Data_RF = read.csv("framingham.csv")
Data_RF = Data_RF[complete.cases(df),]

Data_RF$Sex = as.factor(Data_RF$Sex)
Data_RF$education = as.factor(Data_RF$education)
Data_RF$currentSmoker = as.factor(Data_RF$currentSmoker)
Data_RF$BPMeds = as.factor(Data_RF$BPMeds)
Data_RF$prevalentStroke = as.factor(Data_RF$prevalentStroke)
Data_RF$prevalentHyp = as.factor(Data_RF$prevalentHyp)
Data_RF$diabetes= as.factor(Data_RF$diabetes)
Data_RF$TenYearCHD= as.factor(Data_RF$TenYearCHD)

#labeling my target variable binary data to characters 
# Data_RF[Data_RF$TenYearCHD != 0,"TenYearCHD"] = "Risk_of_CHD"
# Data_RF[Data_RF$TenYearCHD == 0,"TenYearCHD"] = "No_Risk"

#Data partioning
set.seed(123)

train_1= sample(1:nrow(Data_RF),(2/3)*(nrow(Data_RF)))

train_data_1 = Data_RF[train_1,]
test_data_1 = Data_RF[-train_1,]

#predictive model(Random Forest)
# install.packages("randomForest")
library(randomForest)

rftrain = randomForest(TenYearCHD~., data = train_data_1);rftrain

#-------------------------------------------------
#training data results
# Confusion matrix:
#            No_Risk       Risk_of_CHD  class.error
# No_Risk        2057          17       0.008196721
# Risk_of_CHD     345          18       0.950413223
#-------------------------------------------------

#predictive model evaluation with test data


library(caret)

confusionMatrix(predict(rftrain,test_data_1),test_data_1$TenYearCHD, positive = "1")
#-------------------------------------------------
#confusion matrix for test data
#             Reference
#Prediction    No_Risk Risk_of_CHD
#No_Risk        1021         185
#Risk_of_CHD       4           9
#-------------------------------------------------

#sensitivity came off as 0.04632. model was too bad so we do under and over sampling to data
summary(train_data_1$TenYearCHD)
#oversampling
#ROSE - randomly over sampling examples
# install.packages("ROSE")
library(ROSE)

over= ovun.sample(TenYearCHD~.,data = train_data_1, method = "over",N=4148)$data; over
# TenYearCHD  
# No_Risk    :2074  
# Risk_of_CHD: 363  for over sampling N = 2074*2 = 4148


table(over$TenYearCHD)

rfover = randomForest(TenYearCHD~ totChol+sysBP+diaBP+ cigsPerDay+ heartRate+ BMI, data = over);rfover

#After applying the feature selection and over sampling methods
# Confusion matrix:
#     0        1    class.error
# 0 1977      97       0.0467695275
# 1    2     2072      0.0009643202

#without applying the feature selection from PCA
#-------------------------------------------------
#training data results
# Confusion matrix:
#            No_Risk       Risk_of_CHD  
# No_Risk        1997          77       
# Risk_of_CHD     10          2064      
#-------------------------------------------------


confusionMatrix(predict(rfover,test_data_1),test_data_1$TenYearCHD, positive = "1")

#-------------------------------------------------
#confusion matrix for test data
#             Reference
#Prediction    No_Risk Risk_of_CHD
#No_Risk        978        176
#Risk_of_CHD    47           18
#-------------------------------------------------

##sensitivity came off as 0.0924.Bummer!!!! model was not improved .so we try under-sampling the data


#undersampling

under = ovun.sample(TenYearCHD~ totChol+sysBP+diaBP+ cigsPerDay+ heartRate+ BMI, data=train_data_1,method= "under",N= 726)$data 
# TenYearCHD  
# No_Risk    :2074  
# Risk_of_CHD: 363  for over sampling N = 363*2 = 726


table(under$TenYearCHD)

rfunder = randomForest(TenYearCHD~.,data= under)
confusionMatrix(predict(rfunder,test_data_1),test_data_1$TenYearCHD, positive = "1")

##sensitivity came off as 0.5928. model was improved .so we take under sampled data for randomforest model



#modelling in decision tree
fit1= rpart(TenYearCHD~totChol+sysBP+diaBP+ cigsPerDay+ heartRate+ BMI,
            data = under,
            method = "class",
            control= rpart.control(xval= 10,minsplit= 50),
            parms = list(split = "gini") )
fit1


plot(fit1, uniform=TRUE,  # space out the tree evenly
     branch=0.5,         # make elbow type branches
     main="Classification Tree for CHD Prediction",   # title
     margin=0.1)         # leave space so it all fits
text(fit,  use.n=TRUE,   # show numbers for each class
     all=TRUE,           # show data for internal nodes as well
     fancy=FALSE,            # draw ovals and boxes
     pretty=TRUE,           # show split details
     cex=0.8)            # compress fonts to 80%


library(rpart.plot)
prp(fit1, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for CHD Prediction")    
# type can be any values from 0, 1, 2, ...,5, corresponding to different formats
# extra can be any value from 0, 1, 2,..., 11, corresponding to different texts to be displayed
rpart.plot(fit1, type = 1, extra = 1, main="Classification Tree for CHD Prediction")  


CHD_predicted = predict(fit1,test_data_1,type="class");CHD_predicted
CHD_actual = test_data_1$TenYearCHD;CHD_actual

cm_dt = Confusion_matrix = table(CHD_predicted,CHD_actual);Confusion_matrix

pt= prop.table(Confusion_matrix);pt

FPR = cm_dt[2,1]/ sum(cm_dt[2,1]+ cm_dt[1,1]);FPR

FNR = cm_dt[1,2]/ sum(cm_dt[1,2]+ cm_dt[2,2]);FNR

#DT specificity is TNR equal to tn/n 
TNR= DT_Specificity = cm_dt[1,1]/sum(cm_dt[1,1]+cm_dt[2,1]);DT_Specificity 

#DT sensitivity is TPR equal to tp/p

TPR = DT_sensitivity = cm_dt[2,2]/sum(cm_dt[2,2]+cm_dt[1,2]);DT_sensitivity

DT_precision = cm_dt[2,2]/(sum(cm_dt[2,2]+cm_dt[2,1]));DT_precision

#accuracy
accuracy = pt[1,1] + pt[2,2]   # 0.5873

#Sensitivity is 0.6597 but look at precision. its 0.2396. we cannot go with this model. Random forest is better than this model





#-------------------------------------------------
#results till now
#----------------- LOGISTIC REGRESSION-------------
cm= LR_confusionMatrix = table (predicted_data$predicted_CHD, predicted_data$CHD); LR_confusionMatrix


a = LR_Prop_tab= prop.table(LR_confusionMatrix)
LR_Prop_tab

# LRaccuracy
LR_accuracy = LR_Prop_tab[1,1] + LR_Prop_tab[2,2] ;LR_accuracy  # 0.8558534


FPR = cm[2,1]/ sum(cm[2,1]+cm[1,1]);FPR

FNR = cm[1,2]/ sum(cm[1,2]+ cm[2,2]);FNR

#LR specificity is TNR equal to tn/n 
TNR= LR_Specificity = cm[1,1]/sum(cm[1,1]+cm[2,1]);LR_Specificity 

#LR sensitivity is TPR equal to tp/p

TPR = LR_sensitivity = cm[2,2]/sum(cm[2,2]+cm[1,2]);LR_sensitivity

precision = cm[2,2]/(sum(cm[2,2]+cm[2,1]));precision

#------------------RANDOM FOREST--------------------

confusionMatrix(predict(rfover,test_data_1),test_data_1$TenYearCHD, positive = "1")

##sensitivity came off as 0.0724. model was not improved .so we try under sampling the data


#undersampling

under = ovun.sample(TenYearCHD~., data=train_data_1,method= "under",N= 726)$data

table(under$TenYearCHD)

rfunder = randomForest(TenYearCHD~.,data= under)
confusionMatrix(predict(rfunder,test_data_1),test_data_1$TenYearCHD, positive = "1")

##sensitivity came off as 0.6701. model was improved .so we take under sampled data for randomforest model


#-----------------DECISION TREE-------------------------

Confusion_matrix = table(CHD_predicted,CHD_actual);Confusion_matrix

pt= prop.table(Confusion_matrix);pt

FPR = pt[2,1];FPR

FNR = pt[1,2];FNR

#specificity is TNR equal to tn/n 
TNR= Specificity = pt[1,1];Specificity

#sensitivity is TPR equal to tp/p

TPR = sensitivity = a[2,2];sensitivity


#accuracy
accuracy = pt[1,1] + pt[2,2]   # 0.603


#-----------------------------------------------
#                   SVM
#-----------------------------------------------
# install.packages("e1071")
library(e1071)

# split the data into training and test data sets
# set.seed(123)   # for reproducible results


# set kernel='linear', 'polynomial', 'radial basis', or 'sigmoid'
# Fit an SVM model
svm_model <- svm(TenYearCHD ~ ., data = train_data_1, kernel = "radial", scale = TRUE, cost = 10)
#radial or polynomial

# Make predictions on the test set
svm_preds <- predict(svm_model, test_data_1)

# Evaluate the performance of the model
table(svm_preds, test_data_1$TenYearCHD)

summary(svm_model)
summary(test_data_1$TenYearCHD)

#with feature selection
svm_model <- svm(TenYearCHD ~ totChol+sysBP+diaBP+ cigsPerDay+ heartRate+ BMI, data = train_data_1, kernel = "radial", scale = TRUE, cost = 10)
#radial or polynomial

# Make predictions on the test set
svm_preds <- predict(svm_model, test_data_1)

# Evaluate the performance of the model
table(svm_preds, test_data_1$TenYearCHD)

summary(svm_model)
summary(test_data_1$TenYearCHD)


#Through the use of machine learning algorithms and techniques such as dimensional reduction and undersampling,
# this study was able to identify key health factors strongly associated with CHD and develop an accurate classification model to predict 10-year risk.

#The results showed that the random forest algorithm outperformed other classification models in terms of sensitivity, recall, and precision, suggesting that this model could be a valuable tool for identifying patients at risk for CHD.



# Machine_Learning_Project
# Predicting a 10-year risk of future coronary heart disease (CHD)

# Introduction:
Coronary heart disease (CHD) is an inescapable nemesis for those who live with bad lifestyle, but recent research suggests that CHD may follow certain patterns. This study aims to predict the 10-year risk of future CHD using an ongoing cardiovascular study of residents in Framingham, Massachusetts. The dataset, which includes over 4,000 records and 15 attributes, is publicly available on Kaggle.
 
The research question is to identify key health factors strongly associated with CHD and evaluate the accuracy of different classification models, including logistic regression, decision trees, random forests, and support vector machines. The project seeks to develop the most accurate model to help the healthcare system identify patients at risk for CHD, enabling early intervention or prevention methods. 
# Business Problem:
What are the key health factors strongly associated with coronary heart disease (CHD), and which classification model provides the most accurate prediction of the 10-year risk of future CHD? The goal of this project is to assist the healthcare system in identifying patients at risk for CHD, enabling early intervention or prevention methods.

# Data source and description: 
I have taken our data source from Kaggle (https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression?resource=download). The dataset covers history of 4000 residents in Framingham, Massachusetts. The below tables divide 15 variables of the dataset into 4 categories.

# Models used:
1.	Logistic regression 
2.	Decision trees
3.	Random forests
4.	SVM
5.	PCA

# Challenges Faced:
1.	Unbalanced Data:
Our analysis aims to predict the risk of coronary heart disease using a dataset that presents a significant challenge. The dataset is imbalanced, with only 15% of individuals having a risk of coronary heart disease, while the remaining 85% have no such risk. This poses a challenge because standard machine learning models may be biased towards the majority class, leading to lower predictive performance for the minority class. Addressing this issue will require employing techniques such as data resampling or ensemble learning to improve the accuracy and robustness of our predictions.
2.	Biased Dataset:
A biased dataset is a significant challenge in building machine learning (ML) models, as it can lead to inaccurate models that may not perform well on new data. Biases in the dataset can occur due to various reasons, such as under-representation of certain groups, data collection errors, or intentional bias. Biases can cause the model to make incorrect predictions or generalize poorly to new data, resulting in unreliable and potentially harmful outcomes.
3.	Feature Selection
Feature selection involves identifying the most relevant features from the dataset that can help the ML model accurately predict the target variable. It is challenging to select the right features that are relevant to the problem and transform them in a way that improves the model's performance. In addition, selecting too many or too few features can lead to overfitting or underfitting of the model.


# Understanding Data Nature:
1.Used combination of Supervised and Unsupervised Models:

In my project, I encountered a dataset that presented a challenge due to the low correlation between independent and response variables. To overcome this, I used a combination of supervised and unsupervised machine learning models to extract useful insights from the data.

I employed four different supervised models, namely logistic regression, decision tree, random forest, and support vector machine (SVM), to predict the outcome variable based on the independent variables. These models allowed me to detect patterns and relationships in the dataset, and to make accurate predictions based on labeled data.

However, a challenge I encountered was the presence of three highly correlated independent variables, which could have led to multicollinearity issues in the supervised models. To address this, I used Principal Component Analysis (PCA), an unsupervised machine learning technique, to reduce the dimensionality of the data and identify the underlying factors that explain the variance among these variables. This approach helped me to create a new set of independent variables that are linearly uncorrelated, allowing me to avoid the problem of multicollinearity and improve the performance of my supervised models.

In summary, by using a combination of supervised and unsupervised machine learning techniques, I was able to gain a better understanding of the complex relationships within the dataset, and to make more accurate predictions and informed decisions. My approach allowed me to overcome the challenge of low correlation and multicollinearity in the data, which is often encountered in real-world problems. I believe that my findings will be useful for future research in this field and can be used to inform decision-making in similar contexts.

2. Balancing Data:
   
In my project, I encountered an unbalanced dataset, which can lead to biased predictions and reduced model performance. To address this issue, I used two common techniques: oversampling and undersampling.

Oversampling involves increasing the number of instances in the minority class, while undersampling involves reducing the number of instances in the majority class. By using both techniques, I was able to balance the dataset and ensure that the model is trained on an equal number of instances for each class, thus avoiding bias towards the majority class.

After balancing the dataset, I applied the supervised models, namely logistic regression, decision tree, random forest, and support vector machine (SVM), to the dataset. By using the balanced dataset, I ensured that the models were trained on a representative sample of instances for each class, which improved the accuracy of the predictions and the overall performance of the models.

My approach allowed me to overcome the challenge of dealing with unbalanced datasets, which is often encountered in real-world problems. By employing over-sampling and under sampling techniques, I was able to ensure that the models could learn from all the available data while avoiding bias towards the majority class.

# Data Preprocessing:
1. Data Cleaning:

Data cleaning is a crucial step in any data analysis project, as it ensures the accuracy and reliability of the results obtained. In my project, I encountered null values in the dataset, which can cause errors in the analysis and lead to incorrect conclusions.
To address this issue, I used R code to remove the null values from the dataset. This allowed me to ensure that the dataset was complete and that all the necessary information was available for analysis. By removing the null values, I was able to avoid potential errors in the analysis and obtain more accurate results.

2. Nature of Independent Variables:

In my project, I analyzed the nature of the independent variables and their relationship with the response variable. I found that each independent variable was almost normally distributed in relation to the response variable. I visualized this relationship through graphs that demonstrated the normal distribution of each independent variable in relation to the response variable.
By understanding the nature of the independent variables, I was able to gain insights into the underlying patterns and relationships within the dataset. The normal distribution of the independent variables in relation to the response variable suggests a non-linear relationship between the variables, which is useful information for modeling and prediction purposes.
By visualizing this relationship through graphs, I was able to better understand the underlying patterns and relationships within the data, and to make more accurate predictions and informed decisions.

# Model Building:

1.	PCA
The correlation matrix analysis revealed that there is a possibility of reducing the dimensionality of the dataset. To achieve this, I employed the principal component analysis (PCA) technique to extract important features from the dataset. By using PCA, I was able to identify the underlying patterns and correlations within the dataset, and to reduce the number of variables while retaining the important information. PCA can be used to extract the major important predictor for the response out of the existing predictors. This can be used if there is a low correlation between response and independent predictors and the general analysis is unable to capture the general structure of the data to predict the response in Supervised learning.
Logistic Regression:
![1](https://github.com/user-attachments/assets/1d415904-14e6-44b3-8d9a-e78c2d535b6a)
![2](https://github.com/user-attachments/assets/f332461d-6f77-45f4-97b8-84da5632cb47)
![3](https://github.com/user-attachments/assets/41f81332-6588-4b26-973a-5ef419bd85f4)


3.	Decision trees:
Below are the results of Decision Tree with Under sampling and Feature Sampling.
![4](https://github.com/user-attachments/assets/b830da69-2e74-48bc-b45e-1601e9985bc1)

Below is the Classification Diagram of the above Decision Tree Prediction.
![5](https://github.com/user-attachments/assets/140a5557-8daf-4ec7-ab42-21e8895fbbeb)


5.	Random forests:
   
Below are the results of Random Forest technique performed over normal training data.
![6](https://github.com/user-attachments/assets/19df197a-a931-421b-8478-47397d2c4998)


Results of Random Forest technique performed over normal training data. 
![7](https://github.com/user-attachments/assets/6dde60ac-bfe3-4e3a-b7a8-02edf7843554)


Results of Random Forest technique performed over Test data of Oversampled data.
![8](https://github.com/user-attachments/assets/797f13cc-d17f-4103-a73c-295f822cd9c0)

Results of Random Forest technique performed over Test data of Undersampled data
![9](https://github.com/user-attachments/assets/e9111b33-f51b-45a8-8e94-9282890389ae)


4.	SVM:
   
The following are the prediction results using Support Vector Machine technique. Of the all kernels - linear, polynomial, Sigmoid and radial, the radial kernel was better at predicting the risk of heart disease. But through SVM, I was not able to predict whether the people really had a risk of coronary heart disease or not.
![10](https://github.com/user-attachments/assets/d8ff9b4a-6ad3-4755-9f3d-d12900237e59)


# Results and Analysis:
The sensitivity of Random Forest is 0.5928 which is best among the rest of the models taking recall and precision into consideration. The Random Forest model was improved after involving the feature selection and balancing techniques. So, I took under sampled data for random forest model as best Model.

# Conclusion:
Coronary heart disease is a serious health concern that affects a significant portion of the population, but recent research has identified patterns and risk factors that can help to predict future risk.
Using machine learning algorithms and techniques such as dimensional reduction and under sampling, this study was able to identify key health factors strongly associated with CHD and develop an accurate classification model to predict 10-year risk.

The results showed that the random forest algorithm outperformed other classification models in terms of sensitivity, recall, and precision, suggesting that this model could be a valuable tool for identifying patients at risk for CHD.

By providing early intervention or prevention methods, this model has the potential to help improve patient outcomes and reduce the burden of CHD on the healthcare system.

Above all the Models used in random forest have given us better results in terms of sensitivity, recall and precision.


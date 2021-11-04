# Credit_Risk_Analysis

# **Overview** #
This analysis focuses on designing a metric capable of measuring Credit Risk. The data used will be from LendingClub, a peer to peer lending service company. The request for this analysis is done with the goal of measuring the credit risk for different loan and credit products for future credit applications based on the analysis of previous credit products issued. The data provided by LendingClub contains iinformation about their current and previous customers of credit products, characteristics of the product, information acquired when the application was done, and their subsequent behavior of payment history of the credit product. The data was cleaned and prepared for the application of different Machine Learning algorithms. The Machine Learning tools will be done using different libraries for Python via Jupyter Notebook. 

The review of the data showcased different challenges in order to build a credit risk algorithm. 
  - The data is unbalanced with regards of the amount of loans that were paid on time and the loans that were risky or defaulted. The majority of the credit products were not risky and were paid on time. A minority were risky or defaulted. This poses a challenge for Machine Learning algorithms since the data fed to the model that needs to be trained, will be skewed with data that is mostly not risky. That could present different outcomes of the models, either algorithms that will finds most applications as Not Risky because of the skewednesss of the data or will be more aggresive and flag Not Risky applications as Risky. Different Machine Learning approaches will be used in order to find a balanced model.
  - While the data contains sufficient information for an analysis and the design of a model, it only contains Loan Data application for the first quarter of 2019. If a good model is designed with the current data, it would need to be tested with data from other quarters during the year and rule out or adjust depending on changes of seasonality. 


Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, you’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, you’ll use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, you’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. 


# **Results** #
There were different approaches in order to build a Machine Learning model. The first step was to clean and prepare the data before feeding it to different algorithms. The data contaning text or strings were converted to numerical values that the Machine Learning algorithms can process. Once the data was encoded, it was divided between the variables that will feed the algorithm and the variable to receive an output, the Loan Status that would predict whether a Loan was Low Risk or High Risk. 

![1_Original_data_balance](https://user-images.githubusercontent.com/85839235/140198063-ef94235e-0be2-4f2a-bcd7-0445e2ae951d.png)

After cleaning the data, the values available to feed our algorithm was heavily skewed with Low Risk information, as expected. **Logistic Regression** algorithms were applied on 4 different versions of the data. The first version of Logistic Regression was done with balancing the testing data using Random Oversampling. The second Logistic Regression was done with resampling the data with SMOTE Oversampling. The third used a Random Undersampler and lastly, the fourth iteration of Logistic Regression used a Combination Sampling SMOTEEN algorithm. Two other Machine Learning models were used, Banlanced Random Forest Classifier and Easy Ensemble Ada Boost Classifier in order to compare if any model sufficiently predicts risky loans when compared to the test data. 

  ## **Logistic Regression** ##

  - ### **Random Oversampling** ###
This model randomly selects and adds data to the training set until the majority and minority classes are balanced.
![Logistic_Regression_Random_Oversampling](https://user-images.githubusercontent.com/85839235/140242175-77951a56-bc78-4ba3-800e-70f82d2c53a3.png)

**Balance Accuracy Score is 0.65**
The precision for the Low Risk is high at 1, while High Risk is at 0.01. It is reasonable to assume that just creating random sets of data will not compensate enough for the disparity in balance of the data. 


  - ### **SMOTE Oversampling** ###
This model creates synthetic data from the minority class by selecting samples from the minority class and interpolating nearby data.  
![Logistic_Regression_SMOTE_Oversampling](https://user-images.githubusercontent.com/85839235/140242645-735e3d1f-47ae-4b41-bb5d-acc12609813e.png)

**Balance Accuracy Score is 0.63**
The precision for the Low Risk is high at 1, while High Risk is at 0.01. It is reasonable to assume that the creation of synthetic data of the minority class will not compensate enough for the disparity in balance of the data. 


  - ### **Random Undersampling** ###
This model resamples the data in order to decrease the size of hte majority class, resultin in loss of data. 
![Logistic_Regression_Undersampling](https://user-images.githubusercontent.com/85839235/140242907-e425a87d-6923-4702-92c4-2fe2416bf22b.png)

**Balance Accuracy Score is 0.63**
The precision for the Low Risk is high at 1, while High Risk is at 0.01. It is reasonable to assume that just eliminating random sets of data from the majority class will not compensate enough for the disparity in balance of the data. 



  - ### **Combination / SMOTEEN** ###
This model combines aspects of both oversampling and undersampling. It oversamples the minority data with SMOTE and undersamples the majority data with Edited Nearest Neighbors (ENN) algorithm. 
![Logistic_Regression_SMOTEEN](https://user-images.githubusercontent.com/85839235/140240087-4ecf691d-ceb5-407c-a443-b66d7609e61c.png)



  ## **Balanced Random Forest Classifier** ##
This algorithm will sample the data and build several small, simple decision trees. The data is sampled randomly in small portions and a decision tree is created on the small portion of data sampled. Those small trees are then combined. 
![Balanced_Random_Forest_Classifier](https://user-images.githubusercontent.com/85839235/140240097-81a1595d-fabb-4944-bd04-87aba5a5cbad.png)


![Balanced_Random_Forest_Classifier_Sorted_Features](https://user-images.githubusercontent.com/85839235/140240103-5848ff74-f894-4226-aca1-a60622f0c819.png)


  ## **Easy Ensemble / Adaptive Boost** ##
In this algorithm a model is trained then evaluated, after evaluating the errors of the first model, another model is trained. Each new model is build giving more weight to he errors presented on the previous model.  
![Easy_Ensemble_Ada_Boost](https://user-images.githubusercontent.com/85839235/140240111-a7cefa88-0cb7-45df-bac4-f21cb5a6a3b0.png)





Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.


# **Conclusion** #

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

### Reference Links ###

[1️⃣]

# data_analysis
In this repository I would like to present the projects I worked on in the past. 
Once finished this repository will contain analysis involving regression, classification and NLP.

The projects presented are:
- analysing student loans
- predicting customer churn of a telecom company

Installation:
- for creating the same conda envionment please type ```conda env update --name myenv --file environment.yml --prune``` in the terminal



#### Issues:
1. Chi test does not work with SelectKBest for categorical features
2. High variance - Train: 1.0, Val: 0.6
3. What is the optimal features selection technique?


#### Steps:
1. Loading the data
2. EDA
3. Feature Selection
- chi test
- creating stronger features
4. Data Cleaning
- removing outliers > +- 3 stds
- removing duplicates
- removing NaNs
5. Model Training
- categorical features: encoding, chi test (?)
- numerical features: scaling, dropping zero variance, 
6. Model Evaluation
- precision: if model predicts churn, what is the prob. the customer will leave
- recall: % of customers who leave although they were predicted to stay 

Falsly predicting that a customer will leave and offering him an insentive is less costly than failing to notice a customer will leave. Therefore, in this case recall is more important than precision. 
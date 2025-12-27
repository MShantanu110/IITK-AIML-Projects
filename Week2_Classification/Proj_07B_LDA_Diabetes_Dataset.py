from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

DiabetesData = pd.read_csv('pima_indians_diabetes.csv')

#Predict the onset of diabetes based on diagnostic measures
#The objective of the dataset is to diagnostically predict 
#whether or not a patient has diabetes, based on certain 
#diagnostic measurements included in the dataset.

#The datasets consists of several medical predictor variables 
#and one target variable, Outcome. Predictor variables includes 
#the number of pregnancies the patient has had, their BMI, 
#insulin level, age, and so on.

X = DiabetesData.iloc[:,0:8]
Y = DiabetesData.iloc[:,8]
scaler = StandardScaler()
X = scaler.fit_transform(X)

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2,random_state=5)

lda = LinearDiscriminantAnalysis()
lda.fit(Xtrain,Ytrain)
Ypred = lda.predict(X)

ldascore = accuracy_score(lda.predict(Xtest),Ytest)
print('Accuracy score of LDA Classifier is',100*ldascore,'%\n')

cmat = confusion_matrix(lda.predict(Xtest),Ytest)
print('Confusion matrix of LDA is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()
plt.show()
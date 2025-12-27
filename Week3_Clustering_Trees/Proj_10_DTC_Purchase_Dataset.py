import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.metrics import accuracy_score

purchaseData = pd.read_csv('Purchase_Logistic.csv')


#Dataset
#The dataset contains 400 entries for each of the features 
#userId
#gender
#age
#estimatedsalary 

#The target is 
#purchased history 
#The features taken into account are age and estimated salary which are 
#required to predict if the user will purchase a new car (1=Yes, 0=No)

X = purchaseData[['Age', 'EstimatedSalary']] 
Y = purchaseData['Purchased']

# 2. Split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=0)

# 3. Initialize the model
# Using 'entropy' again. You can remove random_state if you want chaos.
cf = DecisionTreeClassifier(criterion='entropy', random_state=0)

# 4. Train the model
cf.fit(Xtrain, Ytrain)

# 5. Predict
Ypred = cf.predict(Xtest)











decPlot = plot_tree(decision_tree=cf, feature_names = ["Age", "Salary"], 
                     class_names =["No", "Yes"] , filled = True , precision = 4, rounded = True)

text_representation = tree.export_text(cf,  feature_names = ["Age","Salary"])
print(text_representation)


cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of DTC is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()

DTCscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of DTC is',100*DTCscore,'%\n')

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score


irisset = datasets.load_iris()

#The Iris dataset was used in R.A. Fisher's classic 1936 paper, 
#The Use of Multiple Measurements in Taxonomic Problems, 
#and can also be found on the UCI Machine Learning Repository.

# It includes three iris species with 50 samples each as well as 
#some properties about each flower.
# The 3 species of iris are
#Iris setosa, Iris virginica and Iris versicolor

#The columns in this dataset are:    
#Id
#SepalLengthCm
#SepalWidthCm
#PetalLengthCm
#PetalWidthCm
#Species



X = irisset.data
Y = irisset.target

# 2. Split into training and testing sets
# I'm using 25% for testing, but you can change test_size to whatever
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=42)

# 3. Initialize the Decision Tree Classifier
# 'criterion' can be 'gini' or 'entropy'. I used entropy just to be fancy.
cf = DecisionTreeClassifier(criterion='entropy', random_state=42)

# 4. Fit (Train) the model
cf.fit(Xtrain, Ytrain)

# 5. Predict the values for the test set
Ypred = cf.predict(Xtest)


decPlot = plot_tree(decision_tree=cf, feature_names = ["sepal_length","sepal_width","petal_length","petal_width"], 
                     class_names =["setosa", "vercicolor", "verginica"] , filled = True , precision = 4, rounded = True)


text_representation = tree.export_text(cf,  feature_names = ["sepal_length","sepal_width","petal_length","petal_width"])
print(text_representation)


cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of DTC is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()

DTCscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of DTC is',100*DTCscore,'%\n')

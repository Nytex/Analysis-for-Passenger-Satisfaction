import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Package for Plotting
import matplotlib.pyplot as plt # library for plotting
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics



#Reading the Dataset
df = pd.read_csv('D:/train.csv')
df_test = pd.read_csv('D:/test.csv')

#Convert the Satisfaction to 0 for neutral of dissatisfied and 1 for satisfied
df['satisfaction_cleaned'] = np.where(df['satisfaction']=="neutral or dissatisfied",0,1)
df_test['satisfaction_cleaned'] = np.where(df_test['satisfaction']=="neutral or dissatisfied",0,1)


######Declare the independent variables and Dependent Variables
##The Independent Variable
x_train = df[['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checking service','Inflight service','Cleanliness']]
x_test = df_test[['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checking service','Inflight service','Cleanliness']]

####The Dependent Variable
y_train = df['satisfaction_cleaned']
y_test = df_test['satisfaction_cleaned']


##Multiple Linear Regression Method
model = linear_model.LinearRegression()
model.fit(x_train,y_train)
print("The prediction accuracy is: {0:2.2f}{1:s}".
      format(model.score(x_test,y_test)*100,"%"))

##Decision Tree
# input the decision tree classifier using "entropy" & train the model
dtree = DecisionTreeClassifier(criterion = 'gini').fit(x_train, y_train)
#The acccuracy will be different for each time

# predict the classes of new, unseen data
predict = dtree.predict(x_test)

print("The prediction accuracy is: {0:2.2f}{1:s}".format(dtree.score(x_test,y_test)*100,"%"))
                                                                     
# Creates a confusion matrix for predicted Item and actual data
cm = confusion_matrix(y_test, predict)
                                                                     
# Transform to dataframe for easier plotting
cm_df = pd.DataFrame(cm, index = ['No','Yes'],
     columns = ['No','Yes'])
                                                                     
# plot the confusion matrix
plt.figure(figsize=(8,8))
ax= sns.heatmap(cm_df, annot=True, fmt='g')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Decision Tree Accuracy:" + str(dtree.score(x_test,y_test)*100))
plt.ylabel('True label')
plt.xlabel('Predicted label')

#Plotting the Tree
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (50,50), dpi=300)
tree.plot_tree(dtree, filled = True);
fig.savefig('imagename.png')



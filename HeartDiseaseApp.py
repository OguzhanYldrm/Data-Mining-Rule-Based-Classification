import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import wittgenstein as lw
from time import perf_counter 

# ------------------------------------------------------------------------------

print("\n********************************************************************\n")
# ------------------------------------------------------------------------------
# PART A -- Read data and select some features
print("(PART A)\n")

HeartDF = pd.read_csv('heart_data.csv', delimiter=',')
print("Dataframe has been read.\n--\n")

df = HeartDF.loc[:,['age','cp','trestbps','thalach','chol','target']]
print("(Filtered Features)\n",df.head())


# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------
#PART B -- replace the age feature to make it better for classification (older person or younger person)
print("(PART B)\n")

df.loc[:,'age'] = df.loc[:,'age'].apply(lambda x : "older person" if x > 55 else "younger person")
print("Age feature is transformed according to 55 limit. (older person ->  >55    younger person ->  <=55)")

# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------
# PART C -- Converting to numerical values
print("(PART C)\n")


def labelEncoder(df):
    attributeArray = list(df.select_dtypes(include=['object']))
    encoder = LabelEncoder()
    for attribute in attributeArray:
        df.loc[:,attribute] = encoder.fit_transform(df.loc[:,attribute])

    return df


labelEncoder(df)

print("(Converted Categorical values to Numerical ones)\n", df.head())


# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART D -- Splitting into train/test set
print("(PART D)\n")

print("Data splitted to training and test sets for being used in Ripper Algorithm and Decision Tree Creation...\n")

train,test = train_test_split(df, test_size=0.2, random_state=123)
print("(Shapes of Train and Test Attributes For Ripper Classifier)")
print("Ripper Train Shape",train.shape)
print("Ripper Test Shape",test.shape)
#*****
x = df.drop('target', axis=1)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
print("\n(Shapes of Train and Test Attributes For Decision Tree Classifier and Rippers Prediction part)")
print("Features Train Shape(X_train)",X_train.shape)
print("Features Test Shape(x_test)",X_test.shape)


# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART E -- Apply RIPPER algorithm to create rules
print("(PART E)\n")

print("Ripper Algorithm is being applied...\n")

#Time start point
ripper_start_time = perf_counter()  

ripper_clf = lw.RIPPER()
ripper_clf.fit(train, class_feat='target',random_state=123)
print("Rule Set\n",ripper_clf.ruleset_)

#Time end point
ripper_stop_time = perf_counter() 

#Running time of algorithm
ripper_elapsed_time = ripper_stop_time-ripper_start_time

# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART F -- Create a decision tree classifier
print("(PART F)\n")

print("Decision Tree is being created...\n")

#Time start point
tree_start_time = perf_counter()  

tree_classifier = DecisionTreeClassifier(criterion="entropy",random_state=123).fit(X_train,y_train)

#Time end point
tree_stop_time = perf_counter() 

#Running time of classification
tree_elapsed_time  = tree_stop_time - tree_start_time

# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

# PART G -- Comparing algorithm running times and AUC performances
print("(PART G)\n")


#Predict the response for test dataset of decision tree
tree_pred  = tree_classifier.predict(X_test)
#Area Under Curve Performance of decision tree
false_positive_rate, true_positive_rate, tree_threshold = metrics.roc_curve(y_test, tree_pred)
tree_auc = metrics.auc(false_positive_rate, true_positive_rate)

#Predict the response for test dataset of ripper
ripper_pred = ripper_clf.predict(X_test)
#Area Under Curve Performance of ripper
fpr, tpr,threshold = metrics.roc_curve(y_test, ripper_pred)
ripper_auc = metrics.auc(fpr, tpr)


print("The Area Under Curve Performace(AUC) of Decision Tree('using entropy') is:",tree_auc)
print("Time elapsed for Decision Tree Classification ",tree_elapsed_time*1000," ms")


print("\nThe Area Under Curve Performace(AUC) of Ripper is:",ripper_auc)
print("Time elapsed for Ripper Classification ",ripper_elapsed_time," s")



# ------------------------------------------------------------------------------
print("\n********************************************************************\n")
# ------------------------------------------------------------------------------

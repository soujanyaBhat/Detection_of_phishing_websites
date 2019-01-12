from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data():
    """
    This helper function loads the dataset saved in the CSV file
    and returns 4 numpy arrays containing the training set inputs
    and labels, and the testing set inputs and labels.
    """

    # Load the training data from the CSV file
    with open('E:\\UTD\\Summer\\Real-time detection of phishing websites\\phishing-detection-master\\dataset.csv', 'rb') as csvfile:
        training_data = pd.read_csv(csvfile, delimiter=',')

    """
    Each row of the CSV file contains the features collected on a website
    as well as whether that website was used for phishing or not.
    We now separate the inputs (features collected on each website)
    from the output labels (whether the website is used for phishing).
    """

    # Extract the inputs from the training data array (all columns but the last one)
    inputs = training_data.values[:,:-1]

    # Extract the outputs from the training data array (last column)
    outputs = training_data.values[:, -1]
    
    #Obtaining feature names
    features=list(training_data.columns.values)

    # Separate the training and testing data
    training_inputs, testing_inputs, training_outputs, testing_outputs= train_test_split(inputs,outputs,test_size=0.2,random_state=42)

    # Return the four arrays
    return training_inputs, training_outputs, testing_inputs, testing_outputs,features

def plot_confusion_matrix(cm,classes,title='Confusion matrix', cmap=plt.cm.Blues):
    y_names=classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(y_names))
    plt.xticks(tick_marks, y_names, rotation=45)
    plt.yticks(tick_marks, y_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
   
    # Load the training data
    train_inputs, train_outputs, test_inputs, test_outputs,features= load_data()

    #Recursive Feature elimination
    clf=LogisticRegression(C=1e5)
    rfe=RFE(clf)
    fit=rfe.fit(train_inputs, train_outputs)
    train_inputs=rfe.transform(train_inputs)
    test_inputs=rfe.transform(test_inputs)
    selected=fit.ranking_
    
    #Printing the Top 15 features
    sel=[]
    for i in range(len(selected)):
        if selected[i] == 1:
            sel.append(features[i])
    print("Selected Features: %s") % sel
    
    #Creating MLP Classifier and training the model
    mlp= MLPClassifier(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(5,2), random_state=1)
    mlp.fit(train_inputs,train_outputs)
    
    # Using the trained classifier to make predictions on the test data
    predictions = mlp.predict(test_inputs)
    
    # Print the accuracy (percentage of phishing websites correctly predicted)
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)
    print "The accuracy of MLP on testing data is: " + str(accuracy)
    
    #Calculating the metrics
    classes=[-1,1]
    print(classification_report(test_outputs, predictions, classes))
    plot_confusion_matrix(confusion_matrix(test_outputs,predictions),classes)
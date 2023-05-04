
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import os
import pickle # saving and loading trained model

# importing required libraries for normalizing data
from sklearn import preprocessing
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,LabelEncoder, MinMaxScaler, OneHotEncoder)
from sklearn.preprocessing import Normalizer, MaxAbsScaler , RobustScaler, PowerTransformer
from sklearn import metrics
from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.model_selection import train_test_split # for splitting the dataset for training and testing
from sklearn.metrics import classification_report # for generating a classification report of model

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import tensorflow as tf

#Import to_categorical depending of your tensor version:
#from tensorflow.keras.utils import to_categorical
#from keras.utils import to_categorical

from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout # importing dense layer
from keras.models import Sequential #importing Sequential layer
from keras.layers import Input
from keras.models import Model

# representation of model layers
from keras.utils.vis_utils import plot_model
import itertools

#Load data
path = os.getcwd() + "/nsl-kdd/KDDTrain+.txt"
names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
dataset = read_csv(path, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:33]
y = array[:,41]

# changing attack labels to their respective attack class
def change_label(df):
  df.label.replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],'Dos',inplace=True)
  df.label.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L',inplace=True)
  df.label.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe',inplace=True)
  df.label.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R',inplace=True)

change_label(dataset)

# creating a dataframe with multi-class labels (Dos,Probe,R2L,U2R,normal)
multi_data = dataset.copy()
multi_label = pd.DataFrame(multi_data.label)

# using standard scaler for normalizing
std_scaler = StandardScaler()
def standardization(df,col):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
  return df

numeric_col = multi_data.select_dtypes(include='number').columns
data = standardization(multi_data,numeric_col)

# label encoding (0,1,2,3,4) multi-class labels (Dos,normal,Probe,R2L,U2R)
le2 = preprocessing.LabelEncoder()
enc_label = multi_label.apply(le2.fit_transform)
multi_data['intrusion'] = enc_label
multi_data.drop(labels=[ 'label'], axis=1, inplace=True)
multi_data = pd.get_dummies(multi_data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")
y_train_multi = multi_data[['intrusion']]
X_train_multi = multi_data.drop(labels=['intrusion'], axis=1)
y_train_multi = LabelBinarizer().fit_transform(y_train_multi)
X_train_multi=np.array(X_train_multi)
y_train_multi=np.array(y_train_multi)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_multi, y_train_multi, test_size=0.20, random_state=43)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

# Spot Check Algorithms
model = Sequential() # initializing model

# input layer and first layer with 50 neurons
model.add(Conv1D(32, 3, padding="same",input_shape = (X_train.shape[1], 1), activation='relu'))
model.add(MaxPool1D(pool_size=(4)))
model.add(Dropout(0.2))
model.add(Conv1D(32, 3, padding="same", activation='relu'))
model.add(MaxPool1D(pool_size=(4)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=50))

# output layer with softmax activation
model.add(Dense(units=5,activation='softmax'))

# defining loss function, optimizer, metrics and then compiling model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train = np.asarray(X_train).astype('float32')
Y_train = np.asarray(Y_train).astype('float32')
# training the model on training dataset
history = model.fit(X_train, Y_train, epochs=100, batch_size=5000,validation_split=0.2)

# predicting target attribute on testing dataset
test_results = model.evaluate(X_validation, Y_validation, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

# testing neural network on test set
path_usage = "KDDTest+.txt"
dataset_usage = read_csv(path_usage, names=names)
services = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i',
            'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001',
            'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn',
            'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
            'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp',
            'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']

# changing attack labels to their respective attack class
change_label(dataset_usage)

# creating a dataframe with multi-class labels (Dos,Probe,R2L,U2R,normal)
multi_data_usage = dataset_usage.copy()
multi_label_usage = pd.DataFrame(multi_data_usage.label)

# using standard scaler for normalizing
std_scaler = StandardScaler()
numeric_col_usage = multi_data_usage.select_dtypes(include='number').columns
data_usage = standardization(multi_data_usage,numeric_col_usage)

# label encoding (0,1,2,3,4) multi-class labels (Dos,normal,Probe,R2L,U2R)
le2_usage = preprocessing.LabelEncoder()
enc_label_usage = multi_label_usage.apply(le2_usage.fit_transform)
multi_data_usage['intrusion'] = enc_label_usage

# droping column label and intrusion and encoding columns protocol_type, service, flag
multi_data_usage.drop(labels=[ 'label'], axis=1, inplace=True)

def search(services, data):
    all_services = set(services)
    all_services_matrix = set(data.to_numpy().flatten())
    missing_services = list(all_services - all_services_matrix)
    total = len(missing_services)
    return total

count = search(services, multi_data_usage)

def add_columns(data, n):
    zeros = np.zeros((data.shape[0], n))
    return np.concatenate((data, zeros), axis=1)

multi_data_usage = pd.get_dummies(multi_data_usage,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")

Y_test_multi_usage = multi_data_usage[['intrusion']]
Y_test_multi_usage = LabelBinarizer().fit_transform(Y_test_multi_usage)
Y_test_multi_usage = np.array(Y_test_multi_usage)

multi_data_usage = multi_data_usage.drop(labels=['intrusion'], axis=1)
multi_data_usage = add_columns(multi_data_usage,count)

X_test_multi_usage = multi_data_usage

# reshaping the matrix and adding a column 1
X_test_usage = np.array(X_test_multi_usage)
X_test_usage = np.reshape(X_test_usage, (X_test_usage.shape[0], X_test_usage.shape[1], 1))

# converting the data type of the NumPy arrays X_train to float32
X_test_usage = np.asarray(X_test_usage).astype('float32')

# Predicted output
y_pred_test = model.predict(X_test_usage, batch_size=500)

# For evaluation
y_pred_argmax_test=(np.argmax(y_pred_test, axis=1))
y_test_argmax=(np.argmax(Y_test_multi_usage, axis=1))

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    Prints and plots the confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Classification report on data set
classes=['normal', 'Dos','Probe', 'R2L','U2R']
print("Classification Report on Data Test \n" , classification_report(y_test_argmax, y_pred_argmax_test, target_names=classes))

cnf_matrix = confusion_matrix(y_test_argmax, y_pred_argmax_test)

# Plot non-normalized confusion matrix
plt.figure(figsize=(7, 7))
plot_confusion_matrix(cnf_matrix, classes=classes,title='Test Confusion matrix')
plt.show()

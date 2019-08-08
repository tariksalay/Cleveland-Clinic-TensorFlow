##############
#Tarik Salay
#CS461
#Program4
#5/10/19
##############

#importing libraries
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd #for data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

#loading the heart.csv file
path = "C:/Users/tarik/PycharmProjects/pr4"
heart_excel = 'C:/Users/tarik/PycharmProjects/pr4/heart.csv'
file_path = os.listdir(path)
heart_data = pd.read_csv(os.path.join("C:/Users/tarik/PycharmProjects/pr4",heart_excel))

#Exploring the data and handling missing values
heart_data.shape  #returns the shape of a tensor
heart_data.columns #feature columns

for i in heart_data.index:
    if (heart_data.loc[i].isnull().sum() != 0):
        print('Missing value at ', i) #to catch which line
print("\n")
print('Here you go!')
print("\n")

#one layer
#all features at first
#separate features and target
data_heart_features = heart_data.loc[:,heart_data.columns!='target']
data_heart_target = heart_data.iloc[:,-1]

X_train_all,X_test_all,y_train_all,y_test_all = train_test_split(data_heart_features,data_heart_target,test_size=0.20,random_state=42)

X_train_all.shape

X_test_all.shape

model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train_all.keys())]),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['accuracy'])

model.summary()

model.fit(X_train_all,y_train_all,epochs=1000)

print(model.evaluate(X_test_all,y_test_all))

#Using four features only at second
data_features = heart_data.loc[:,['cp','slope','exang','thal']]
data_target = heart_data.iloc[:,-1]

X_train_four,X_test_four,y_train_four,y_test_four = train_test_split(data_features,data_target,test_size=0.20,random_state=42)

model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train_four.keys())]),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['accuracy'])

model.fit(X_train_four,y_train_four,epochs=1000) #steps

print(model.evaluate(X_test_four,y_test_four)) #output
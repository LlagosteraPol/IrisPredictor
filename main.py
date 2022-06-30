import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

COLUMN_NAMES = [
        'SepalLength',
        'SepalWidth',
        'PetalLength',
        'PetalWidth',
        'Species'
        ]

data = pd.read_csv('Data/iris_data_categorical.csv', names=COLUMN_NAMES, header=0)

data['Species'] = data['Species'].astype("category")

# Check levels of dependency between some of the features
corrMatt = data[["SepalLength","SepalWidth","PetalLength","PetalWidth","Species"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
#plt.show()  # Uncoment to show heatmap

output_data = data["Species"]
input_data = data.drop("Species",axis=1)
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.3, random_state=42)

class IrisClassifier(tf.keras.Model):
        def __init__(self):
                super(IrisClassifier, self).__init__()
                self.layer1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
                self.layer2 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
                self.outputLayer = tf.keras.layers.Dense(3, activation=tf.nn.softmax)

        def call(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return self.outputLayer(x)


# Create model
model = IrisClassifier()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=300, batch_size=10)

# Model evaluation
scores = model.evaluate(X_test, y_test)
print("\nAccuracy: %.2f%%" % (scores[1]*100))

# Model predictions
prediction = model.predict(X_test)
prediction1 = pd.DataFrame({'IRIS1':prediction[:,0],'IRIS2':prediction[:,1], 'IRIS3':prediction[:,2]})
prediction1.round(decimals=4).head()
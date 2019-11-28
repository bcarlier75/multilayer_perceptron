import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


data = pd.read_csv('data.csv', delimiter=',', header=None)
x = data.iloc[:, 2:]
y = data.iloc[:, 1]
std_scaler = StandardScaler()
scaled_features = std_scaler.fit_transform(x)
x_preprocessed = pd.DataFrame(scaled_features, index=x.index, columns=x.columns)
y_preprocessed = np.where(y == 'M', 1, 0)   # 1 means Malign and 0 means Benign
x_train, x_test, y_train, y_test = train_test_split(x_preprocessed, y_preprocessed, test_size=0.25, random_state=42)

dnnClassifierModel = tf.estimator.DNNClassifier(hidden_units=[30, 30],
                                                feature_columns=x.columns,
                                                n_classes=2,
                                                activation_fn=tf.nn.tanh,
                                                optimizer=tf.train.AdamOptimizer(1e-4),
                                                dropout=0.1)

train_input_fn = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
dnnClassifierModel.train(input_fn=train_input_fn, steps=1000)


test_input_fn = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, num_epochs=1, shuffle=False)
accuracy_score = dnnClassifierModel.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))

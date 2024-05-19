from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_breast_cancer
import pandas as pd
from logisticregression import LogisticRegression
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, Y = load_breast_cancer(return_X_y=True)
columns = data.feature_names
data_df = pd.DataFrame(data=X, columns=columns)
data_df["target"] = pd.Series(Y)


x_train,  x_test, y_train, y_test = train_test_split(X, Y,test_size=0.2,shuffle=True)

lr = LogisticRegression(lr=0.0001, n_iters=1000)
print(y_train.shape)
lr.fit(x_train,y_train)

pred = lr.predict(x_test)

accuracy = accuracy_score(y_test,pred)
print("Test accuracy: {0:.3f}".format(accuracy))

print(confusion_matrix(y_test,pred))

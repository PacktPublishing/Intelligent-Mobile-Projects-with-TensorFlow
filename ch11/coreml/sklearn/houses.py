from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

import pandas as pd
import sklearn.model_selection as ms


# Downloaded from: https://wiki.csc.calpoly.edu/datasets/wiki/Houses
# Load data
data = pd.read_csv('houses.csv')

#model.fit(data[["Bedrooms", "Bathrooms", "Size"]], data["Price"])
X, y = data.iloc[:, 3:6], data.iloc[:, 2]

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.25)

lr = LinearRegression()
lr.fit(X_train, y_train)
#print(lr.score(X_test, y_test))

X_new = [[ 3, 2, 1560.0],
         [3,  2,  1680],
         [5, 3, 2120]]

print(lr.predict(X_new))

svm = LinearSVR(random_state=42)
svm.fit(X_train, y_train)
#print(svm.score(X_test, y_test))

print(svm.predict(X_new))


 # Convert and save the scikit-learn model
import coremltools
coreml_model = coremltools.converters.sklearn.convert(lr, ["Bedrooms", "Bathrooms", "Size"], "Price")
coreml_model.save("HouseLR.mlmodel")

coreml_model = coremltools.converters.sklearn.convert(svm, ["Bedrooms", "Bathrooms", "Size"], "Price")
coreml_model.save("HouseSVM.mlmodel")

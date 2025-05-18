import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

path = 'auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

dataset = pd.read_csv(path, names=column_names, na_values='?', comment='\t', sep = ' ', skipinitialspace=True)

print("shape", dataset.shape)
print(dataset.isna().sum())

dataset = dataset.dropna()

names = column_names
names.remove("MPG")

for var in names:
  plt.figure()
  sns.regplot(x=var, y="MPG", data=dataset)
plt.show()

correlations = dataset.corr()
sns.heatmap(correlations)
plt.show()

y = dataset["MPG"]
x = dataset[names]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

print("b = ",regressor.intercept_)
df_coefs = pd.DataFrame(data = regressor.coef_, index=x.columns, columns=["Coef"])
print(df_coefs)

#dataframe
preds = regressor.predict(x_test)
df_preds = pd.DataFrame({"Actual": y_test.squeeze(), "Predicted": preds.squeeze()})
print(df_preds)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)

print("Mean ab", mae)
print("Mean sq", mse)
raiz = np.sqrt(mse)
print("Raiz", raiz)



print("score", regressor.score(x_train, y_train))
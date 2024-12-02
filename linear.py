import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
X = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)
y = [7,9,13,17.5,18]
#建立模型
model = LinearRegression()
model.fit(X,y)
test_pizza = np.array([[12]])
predicted_price = model.predict(test_pizza)[0]
print(predicted_price)
plt.figure()
plt.title("pizza price plotted against diameter")
plt.xlabel("Diameter in inches")
plt.ylabel("price in dollars")

plt.plot(X,y,'k.')

plt.axis([0,25,0,25])
plt.grid(True)
plt.show()


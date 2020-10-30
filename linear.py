
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn import metrics

X=np.array([[1,x]for x in range(1000)])
y=2*X[:,1]+1+np.linspace(1,10000,1000)*10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=11)

lr=LinearRegression()
lr.fit(X_train,y_train)
plt.title("Train Data")
plt.grid("on")
plt.plot(X_train[:,1],y_train)
plt.savefig('data/train.png')
plt.savefig('train.png')

prediction=lr.predict(X_test)
plt.scatter(y_test,prediction)
metrics.mean_squared_error(y_test,prediction)
plt.title("Test Data")
plt.grid("on")
plt.plot(X_train[:,1],y_train,"b*")
plt.plot(X_test[:,1],prediction,"r*")
plt.savefig('data/test.png')
plt.savefig('test.png')


plt.show()

import matplotlib.pyplot as plt
import numpy as np
import random

def multinorm_translate(_mu,_sigma):
    dim = _sigma.shape[0]
    #compute _sigma^(-1/2)
    eigva, eigvc = np.linalg.eig(_sigma)
    eigvc = np.matrix(eigvc)
    eigva_half = np.diag(eigva**0.5)
    sigma_half = eigvc.T * eigva_half * eigvc

    #normalized gaussion
    point = [random.gauss(0,1) for i in range(dim)]
    #transform
    point = point * sigma_half + _mu

    return np.array(point)[0].tolist()

mu = np.array([5,5])
sigma = np.array([[4,0],[0,3]])
PointNum = 300
features = []
for i in range(PointNum):
    point = multinorm_translate(mu,sigma)
    features.append(point)

X = []
Y = []
for item in features:
    X.append(item[0])
    Y.append(item[1])

plt.scatter(X, Y , 30, color ='r', marker = 'o')
plt.show()

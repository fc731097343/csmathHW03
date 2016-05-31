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

def gauss_pdf(point, mu, sigma):
    d = point.shape[0]
    p = 1.0/((2*np.pi)**(d*0.5) * np.linalg.det(sigma)**(0.5)) * np.exp(-0.5 * (np.dot(np.dot((point-mu), np.linalg.inv(sigma)), (point-mu).T)))
    return p

def Expect(data, mu, sigma, w):
    reslist=[]
    for i in range(len(data)):
        plist = []
        for j in range(len(w)):
            p = w[j] * gauss_pdf(data[i],mu[j],sigma[j]);
            plist.append(p.tolist()[0][0])
        psum = sum(plist)
        plist = [p/psum for p in plist]
        reslist.append(plist)
    return reslist

def Maximize(pl, data, mu):
    d = data.shape[1]
    new_w = []
    new_mu = []
    new_sigma =[]
    for i in range(len(mu)):
        p = 0
        _mu = np.zeros([1,dim])
        _sigma =  np.zeros([dim,dim])
        for j in range(len(data)):
            p += pl[j][i]
            _mu += np.array(pl[j][i]) * data[j]
            _x = np.matrix(data[j]-mu[i])
            _sigma += np.array(_x.T * _x) * pl[j][i]

        new_w.append(p/len(data))
        new_mu.append(_mu/p)
        new_sigma.append(_sigma/p)

    return new_w,new_mu,new_sigma


K = 2

mu1 = np.array([5,5])
sigma1 = np.array([[4,0],[0,4]])
PointNum1 = 500

features = []
for i in range(PointNum1):
    point = multinorm_translate(mu1,sigma1)
    features.append(point)

mu2 = np.array([-3,-2])
sigma2 = np.array([[2,0],[0,5]])
PointNum2 = 500

for i in range(PointNum2):
    point = multinorm_translate(mu2,sigma2)
    features.append(point)

features = np.matrix(features)
mean = sum(features) / len(features)

print mean
dim = sigma1.shape[0]
sigma0 = np.zeros([dim,dim])
for i in range(features.shape[0]):
    tem_x = features[i,:] - mean
    sigma0 += np.array(tem_x.T * tem_x)
sigma0 /= 1.0 * features.shape[0]

Weight = [1.0/K for i in range(K)]
Mu = [mean + random.uniform(-6, 6) for i in range(K)]
Sigma = [sigma0 for i in range(K)]

print Mu
print Sigma


iters = 0
delta = 100
while(delta > 0.0001):
    iters += 1

    w_old = Weight
    mu_old = Mu
    sigma_old = Sigma
    Plist = Expect(features, Mu, Sigma, Weight)
    [Weight, Mu, Sigma] = Maximize(Plist, features, Mu)

    delta = sum(sum(sum(abs(np.array(Mu)- np.array(mu_old)))))
    print "iteration %d, delta = %f" % (iters, delta)

print "Weight of %d components: " % (K), Weight
print "Mu of %d components: " % (K), Mu
print "Sigma of %d components: " % (K), Sigma


import numpy as np
from numpy import linalg
from matplotlib  import cm
import matplotlib.pyplot as plt
from sympy import degree
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel

train = np.loadtxt('train.csv', delimiter=',')
test = np.loadtxt('test.csv', delimiter=',')
m = len(train)
n = len(test)

x = train[:,0]
y = train[:,1]
z = train[:,2]

X = test[:,0]
Y = test[:,1]
Z = test[:,2]

## Train scatter 2d plot
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111)
# ax.scatter(x,y,c=z, alpha=0.6, marker = 'o', cmap = 'RdBu')
# plt.show()

def sign(x):
    if x>0:
        return 1
    else:
        return -1

#Initialization
alpha_rbf = np.zeros(m)
alpha_poly = np.zeros(m)
y_hat_train_rbf = np.zeros(m)
y_hat_train_poly = np.zeros(m)

#Choose kernel function
rbf = False
polynomial = True

losses = 0

#Applying on train data
if rbf:
    kernel =  rbf_kernel(train[:,:2], gamma = 0.1)
    T = 50
    l_rbf = []
    for t in range(T):
        for i in range(m):
            y_hat = sign(np.dot(alpha_rbf, kernel[:,i]))
            y_hat_train_rbf[i] = y_hat
            if (z[i] != y_hat): losses += 1
            alpha_rbf[i] += (0.5 * (z[i]-y_hat))
        l_rbf.append(losses/m)
        train_losses_rbf = losses
        losses = 0

if polynomial:
    kernel = polynomial_kernel(train[:,:2], degree=2, gamma=1, coef0=1)
    T = 27
    l_poly = []
    for t in range(T):
        for i in range(m):
            y_hat = sign(np.dot(alpha_poly, kernel[:,i]))
            y_hat_train_poly[i] = y_hat
            if (z[i] != y_hat): losses += 1
            alpha_poly[i] += (0.5 * (z[i]-y_hat))
        l_poly.append(losses/m)
        train_losses_poly = losses
        losses = 0

#Applying on test data

test_kernel_rbf = rbf_kernel(train[:,:2], test[:,:2], gamma=0.1)
test_kernel_poly = polynomial_kernel(train[:,:2], test[:,:2], degree=2, gamma=1, coef0=1)

y_hat_test_rbf = []
y_hat_test_poly = []
misclassified_rbf = 0
misclassified_poly = 0

for j in range(n):
    y_hat_test_rbf.append(sign(np.dot(alpha_rbf, test_kernel_rbf[:,j])))
    y_hat_test_poly.append(sign(np.dot(alpha_poly, test_kernel_poly[:,j])))
    if (test[j,2] != y_hat_test_rbf[j]): misclassified_rbf += 1
    if (test[j,2] != y_hat_test_poly[j]): misclassified_poly += 1


if rbf:
 print('RBF Train errors:', train_losses_rbf)
 print('RBF Test errors:', misclassified_rbf)
 plt.plot(l_rbf)
 plt.ylabel('losses')
 plt.xlabel('Epochs')
 plt.legend(['RBF', 'Polynomial'])

if polynomial:
 print('Polynomial Train errors:', train_losses_poly)
 print('Polynomial Test errors:', misclassified_poly)
 plt.plot(l_poly)
 plt.ylabel('losses')
 plt.xlabel('Epochs')
 plt.legend(['RBF', 'Polynomial'])


fig = plt.figure(figsize=(6,6))
if polynomial:
 ax1 = fig.add_subplot(111)
 ax1.scatter(x,y,c=y_hat_train_poly, alpha=0.6, marker = 'o', cmap = 'RdBu')
 ax1.scatter(X,Y,c=y_hat_test_poly, alpha=0.6, marker = '*', cmap = 'RdBu')
 ax1.set_title('Polynomial 2-degree')
if rbf:
 ax1.scatter(x,y,c=y_hat_train_rbf, alpha=0.6, marker = 'o', cmap = 'RdBu')
 ax1.scatter(X,Y,c=y_hat_test_rbf, alpha=0.6, marker = '*', cmap = 'RdBu')
 ax1.set_title('RBF Gamma=0.1')

plt.show()


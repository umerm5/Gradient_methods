#####Gradient Methods Example
#####Muhammad Umer
#####GIT_ROOT
#####Import different packages
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.misc import derivative


#####Define function and its derivatives in symbolic form
x= sp.symbols('x')
def f(x):
    return ((x**2)/10 - 2*sp.sin(x))
def d(x):
    return derivative(f,x)
def df(x):
    return x/5 - 2*sp.cos(x)
def ddf(x):
    return 1/5 + 2*sp.sin(x)


#####Simple Gradient Descent Function with fixed step size
def GD(x0, alpha, eps,max_iter):
    iter = 0
    y = x0
    y_new = np.zeros((max_iter,1))
    y_new[iter,] = y
    iter += 1
    y = y - alpha * df(y)
    y_new[iter,] = y
    while (np.abs(y_new[iter,] - y_new[iter-1,]) > eps and iter < max_iter):
        iter += 1
        if (iter == max_iter):
            break
        y = y - alpha* df(y)
        y_new[iter,] = y
    return y,iter,y_new

#####Steepest Gradient Descent Function with adaptive step size
def SGD(x0, alpha, eps,max_iter,gamma):
    iter = 0
    y = x0
    y_new = np.zeros((max_iter,1))
    y_new[iter,] = y
    iter += 1
    alpha = gamma * alpha
    y = y - alpha * df(y)
    y_new[iter,] = y
    while(np.abs(y_new[iter,] - y_new[iter-1,]) > eps and iter < max_iter) :
        iter += 1
        if(iter == max_iter):
            break
        y = y - alpha* df(y)
        y_new[iter,] = y
        alpha = gamma * alpha
    return y,iter,y_new

#####Steepest Gradient Descent Function with proposed adaptive step size
def SGD_proposed(x0, alpha, eps,max_iter,gamma):
    iter = 0
    y = x0
    y_new = np.zeros((max_iter,1))
    y_new[iter,] = y
    iter += 1
    alpha = (gamma**(iter)) * alpha
    y = y - alpha * df(y)
    y_new[iter,] = y
    while(np.abs(y_new[iter,] - y_new[iter-1,]) > eps and iter < max_iter) :
        iter += 1
        if(iter == max_iter):
            break
        y = y - alpha* df(y)
        y_new[iter,] = y
        alpha = (gamma**(iter)) * alpha
    return y,iter,y_new

#####Steepest Gradient Descent Function with Armijo rule reduction step size
def SGD_w_armijo(x0, s, sig, beta, max_iter):
    iter = 0
    y = x0
    alpha = s
    y_new = np.zeros((max_iter,1))
    y_new[iter,] = y
    for i in range(1,max_iter):
        alpha = s * (beta ** (i))
        y = y - alpha * df(y)
        print(y)
        y_new[i,] = y
        if ((f(y_new[i-1,]) - f(y_new[i,])) <= (1*sig *s *(beta ** (i)) * (df(y_new[i-1,]))**2)):
            break
    return y, i, y_new

#####Newton Method
def NM(x0, eps, max_iter):
        iter = 0
        y = x0
        y_new = np.zeros((max_iter, 1))
        y_new[iter,] = y
        iter += 1
        y = y - df(y)/ddf(y)
        y_new[iter,] = y
        while (np.abs(y_new[iter,] - y_new[iter - 1,]) > eps and iter < max_iter):
            iter += 1
            if (iter == max_iter):
                break
            y = y - df(y)/ddf(y)
            y_new[iter,] = y
        return y, iter, y_new

#####Call Gradient Descent with fixed step size
res,iter,y_new = GD(x0=0.5,alpha=1,eps=1e-5,max_iter  = 1000)
#####Call Gradient Descent with adaptive step size
res_sgd,iter_sgd,y_new_sgd = SGD(x0=0.5,alpha=1,eps=1e-5,max_iter  = 1000, gamma = 0.5)
#####Call Gradient Descent with proposed adaptive step size
res_sgd_p,iter_sgd_p,y_new_sgd_p = SGD_proposed(x0=0.5,alpha=1,eps=1e-5,max_iter  = 1000, gamma = 0.95)
#####Call Gradient Descent with armijo rule reduced step size
res_sgd_w_armijo,iter_sgd_w_armijo,y_new_sgd_w_armijo = SGD_w_armijo(x0=0.5,s=1, sig=1e-5, beta = 0.9, max_iter = 1000)
#####Call Newton Method
res_NM,iter_NM,y_new_NM = NM(x0=-6,eps=1e-5,max_iter  = 1000)

#####Print Results for Gradient Descent Method with fixed step size
print(res)
print(iter)
print(y_new[0:iter,])

#####Print Results for Gradient Descent Method with adaptive step size
print(res_sgd)
print(iter_sgd)
print(y_new_sgd[0:iter_sgd,])

#####Print Results for Gradient Descent Method with proposed step size
print(res_sgd_p)
print(iter_sgd_p)
print(y_new_sgd_p[0:iter_sgd_p,])

#####Print Results for Gradient Descent Method with Armijo Rule
print(res_sgd_w_armijo)
print(iter_sgd_w_armijo)
print(y_new_sgd_w_armijo[0:iter_sgd_w_armijo,])

#####Print Results for Newton Method
print(res_NM)
print(iter_NM)
print(y_new_NM[0:iter_NM,])


#####Plotting Results

y = np.linspace(-10,10)
f_orig = (np.power(y, 2))/10 - 2*np.sin(y)
plt.plot(y,f_orig)
f1 = (np.power(y_new[0:iter,], 2))/10 - 2*np.sin(y_new[0:iter,])
plt.plot(y_new[0:iter,],f1,'k--o')
plt.title('steepest descent with x0=0.5 and alpha=1')
plt.show()
plt.figure()
plt.plot(y,f_orig)
f1_sgd = (np.power(y_new_sgd[0:iter_sgd,], 2))/10 - 2*np.sin(y_new_sgd[0:iter_sgd,])
plt.plot(y_new_sgd[0:iter_sgd,],f1_sgd,'k--o')
plt.title('steepest descent with a forgetting factor rule x0=0.5, gamma = 0.5, and alpha=1')
plt.show()
plt.figure()
plt.plot(y,f_orig)
f1_sgd_p = (np.power(y_new_sgd_p[0:iter_sgd_p,], 2))/10 - 2*np.sin(y_new_sgd_p[0:iter_sgd_p,])
plt.plot(y_new_sgd_p[0:iter_sgd_p,],f1_sgd_p,'k--o')
plt.title('steepest descent with a proposed forgetting factor rule x0=0.5, gamma = 0.95, and alpha=1')
plt.show()
plt.figure()
plt.plot(y,f_orig)
f1_sgd_w_armijo = (np.power(y_new_sgd_w_armijo[0:iter_sgd_w_armijo,], 2))/10 - 2*np.sin(y_new_sgd_w_armijo[0:iter_sgd_w_armijo,])
plt.plot(y_new_sgd_w_armijo[0:iter_sgd_w_armijo,],f1_sgd_w_armijo,'k--o')
plt.show()
plt.figure()
plt.plot(y,f_orig)
f1_NM = (np.power(y_new_NM[0:iter_NM,], 2))/10 - 2*np.sin(y_new_NM[0:iter_NM,])
plt.plot(y_new_NM[0:iter_NM,],f1_NM,'k--o')
plt.title('Newton Method with x0=-6')
plt.show()
import numpy as np
import matplotlib.pyplot as plt 

def f(x):
    return (np.sin(x/5)*np.exp(x/10)+5*np.exp(-x/2))

def pol(x,tensor):
    res = 0
    for i in range(len(tensor)):
        res = res + (x**i) * tensor[i]
    return res


def tensor(n):
    xt = np.linspace(1,15,n+1)

    yt = f(xt)

    B = yt
    A = np.zeros ((n+1, n+1))
    for i in  range(n+1):
        for j in range(n+1):
            A[j, i] = xt[j]**i
    ten = np.linalg.solve(A,B)
    return ten


x = np.linspace(1,15,1000)


n = 3
xt = np.linspace(1,15,n+1)

yt = f(xt)

B = yt
A = np.zeros ((n+1, n+1))
for i in  range(n+1):
    for j in range(n+1):
        A[j, i] = xt[j]**i
A


ten = np.linalg.solve(A,B)

plt.plot(x,f(x))
plt.plot(x,pol(x,ten))
plt.plot(xt,yt,'.')
plt.show()
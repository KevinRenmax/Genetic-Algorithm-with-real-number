
3第二题：
# import numpy as np
#
# f = open(r'D:\py1451343\Sam001.txt','r')
# lines = f.readlines()
# data1 = []
# for line in lines:
#     a=line.split(",")
#     onerow = list(map(float,a))
#     data1.append(round(onerow[1],5))
#
# f = open(r'D:\py1451343\Sam002.txt','r')
# lines = f.readlines()
# data2 = []
# for line in lines:
#     a=line.split(",")
#     onerow = list(map(float,a))
#     data2.append(round(onerow[1],5))
#
# f = open(r'D:\py1451343\Sam003.txt','r')
# lines = f.readlines()
# data3 = []
# for line in lines:
#     a=line.split(",")
#     onerow = list(map(float,a))
#     data3.append(round(onerow[1],5))
#
# f = open(r'D:\py1451343\Sam004.txt','r')
# lines = f.readlines()
# data4 = []
# for line in lines:
#     a=line.split(",")
#     onerow = list(map(float,a))
#     data4.append(round(onerow[1],5))
#
# f = open(r'D:\py1451343\Sam005.txt','r')
# lines = f.readlines()
# data5 = []
# for line in lines:
#     a=line.split(",")
#     onerow = list(map(float,a))
#     data5.append(round(onerow[1],5))
#
# aa = np.array([data1,data2,data3,data4,data5])
# aa = aa.T
# np.savetxt('D:\py1451343\\sam_merge.txt',aa,fmt='%10.5f', delimiter='\t', newline='\r\n')



#第三题
# from sklearn.cross_decomposition import PLSRegression
# import numpy as np
#
# for j in range(1,11):
#     pls = PLSRegression(n_components=j,scale=False)
#
#     X = np.loadtxt(r"D:\py1451343\\sam_merge.txt")
#     X = X.T
#     y = np.array([16.579,13.611,14.334,12.865,12.992])
#
#     pls.fit(X,y)
#     yhat = pls.predict(X)
#     err = []
#     for i in range(5):
#         err.append(float(y[i] - yhat[i])/(float(y[i])*100))
#     sum =0
#     for i in err:
#         sum += abs(i)
#     print("i = " )
#     print(j)
#     print("err = ")
#     print(sum)


#第四题：
import matplotlib.mlab as mlab
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

delta = 0.5
pi = np.pi
x = np.arange(-pi, pi, delta)
y = np.arange(-pi, pi, delta)
X, Y = np.meshgrid(x, y)
z = np.cross(x,y)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, z, cmap=cm.jet, linewidth=0.2)
plt.show()

plt.figure()
CS = plt.contour(X, Y, z,10,colors='k') # 制作等高线，横砍10刀
plt.clabel(CS, inline=1, fontsize=9) #inline控制画标签，移除标签下的线
plt.title('Simplest default with labels')
plt.show()


#第五题：
import math
import numpy as np
from scipy.stats import f



import numpy as np
class PCA:
    def __init__(self, X):
        self.X=X
    def SVDdecompose(self):
        B = np.linalg.svd(self.X,full_matrices=False)
        U=B[0]
        lamda=B[1]
        self.P = B[2].T
        i=len(lamda)
        S=np.zeros ((i,i))
        S[:i,:i]=np.diag (lamda)
        self.T = np.dot (U,S)
        compare=[]
        for i in range(len(lamda)-1):
            temp = lamda[i]/lamda[i+1]
            compare.append(temp)
        return np.array(compare)

# a = np.loadtxt(r"D:\py1451343\sincos.txt")
# pca = PCA(a)
# compare = pca.SVDdecompose()
# print(compare.round(5))





































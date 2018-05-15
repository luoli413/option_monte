from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as ln
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def put_payoff(k,s):
    if k<s:
        return 0.0
    else:
        return k-s

class matrix_A(object):
    def __init__(self,sigma,r,s_step,t_step):
        self.sigma =sigma
        self.r =r
        self.s_step = s_step
        self.t_step = t_step
    def a(self,n):
        return -0.5*self.t_step*self.sigma**2*(n**2)
    def b(self,n):
        return 1+self.t_step*(self.r+self.r*n+self.sigma**2*(n**2))
    def c(self,n):
        return self.t_step*(-r*n-0.5*self.sigma**2*(n**2))
    def generate_A(self,size):
        A = np.zeros((size,size))
        for i in range(0,size):
            if i== 0:
                A[i,i:(i+2)] = np.array([self.b(i+1),self.c(i+1)]).reshape(2)
            elif size - i == 1:
                A[i,(i-1):] = np.array([self.a(i+1),self.b(i+1)]).reshape(2)
            else:
                A[i,(i-1):(i+2)] = np.array([self.a(i+1),self.b(i+1),self.c(i+1)]).reshape(3)
        return A

class implicit_scheme(object):

    def __init__(self,K,r,sigma,Smax,T,):
        self.K = K
        self.r =r
        self.sigma = sigma
        self.Smax = Smax
        self.T = T

    def scheme(self,sk,tn):
        vpay_off = np.vectorize(put_payoff)
        sk = int(sk)
        tn = int(tn)
        t_step = self.T / tn
        s_step = self.Smax / sk

    # ===initialize our price and boundary meshes========================
        mesh = np.zeros((sk+1,tn+1))
        stop_mesh = np.zeros((sk+1,tn+1))

    # ======= boundary condition =========
    #   when s = smax
        mesh[sk,:] = np.zeros((1,tn+1))
        # s =0
        mesh[0,:] = np.exp(-r*t_step*np.linspace(0,tn,num=tn+1))*self.K
        # t = 0
        vs = s_step * np.linspace(0,sk,num = sk+1)
        vk = self.K * np.ones(sk+1)
        mesh[:,0] = vpay_off(vk,vs).reshape(sk+1)
    # stop_mesh boundary condition
        stop_mesh[:,0] = vk
    # create an object of matrix_A
        As = matrix_A(self.sigma,self.r,s_step,t_step)
        vs = s_step * np.linspace(1, sk - 1, num = sk - 1)
        vk = K * np.ones(sk - 1)
        vr = vpay_off(vk, vs).reshape((sk - 1, 1))
    #     initial free boundary
        boundary = np.zeros((tn+1,))
        boundary[0] = stop_mesh[0,0]
    # ====================Solving PDE==================================
        for n in range(1,tn+1):
            extra = np.zeros((sk - 1, 1))
            B = mesh[1:sk,n-1]
            extra[0,0] = -As.a(1)*K*np.exp(-r*(n+1)*t_step)
            mesh[1:sk,n] = np.linalg.solve(As.generate_A(sk-1),B.reshape(sk-1,1)+extra).reshape(sk-1)
            # American option,we need compare the payoff and PDE
            temp = np.where(mesh[1:sk, n] < vr.reshape(sk - 1))[0]
            indexing = temp.reshape(len(temp),1)
            mesh[1:sk,n][indexing] = vr[indexing,0]
            stop_mesh[1:sk,n][indexing] = vs.reshape(sk-1,1)[indexing,0]
            boundary[n] = np.max(stop_mesh[1:sk,n])
        # pd.DataFrame(mesh).to_csv('mesh.csv')
        # pd.DataFrame(stop_mesh).to_csv('boundary.csv')
        return mesh,boundary

    def scheme_plot(self,mesh,boudary,sk,tn):
        sk = int(sk)
        tn = int(tn)
        t_step = self.T / tn
        s_step = self.Smax / sk

        # Make data.
        X = np.arange(0, self.T + t_step, t_step)
        Y = np.arange(0, self.Smax + s_step, s_step)
        X, Y = np.meshgrid(X, Y)

        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_surface(X, Y, mesh, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(0, self.K)
        ax.set_ylim(0, self.Smax)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlabel('Time to expiration')
        ax.set_ylabel('Spot price')
        ax.set_zlabel('American put option price')
        ax.set_title('Mesh: Time points = '+ str(tn) +', Spot points = ' + str(sk))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # second
        plt.figure(2)
        plt.plot(t_step*np.arange(0,tn+1,step = 1),boudary,)
        plt.xlabel('Time to expiration')
        plt.ylabel('Boundary')
        plt.show()

if __name__ == "__main__":
    # initialization
    r = 0.01
    sigma = 0.2
    T = 1.0
    K = 100
    Smax = 200.0
    start = 16
    s_pow = 2
    t_pow = 4
    error = 0
    pre = []
    num = 5
    AmerPut = implicit_scheme(K, r, sigma, Smax, T)
    # sk = 512
    # tn = 512
    # # for i in list(16*np.power(2,np.linspace(0,10,num = 11))):
    # #
    # #     # print('tn =',i)
    # #     mesh = scheme(r,sigma,Smax,sk,T,i)
    # #     if pre !=[]:
    # #         # find the maxmium of difference vector's norm
    # #         error = np.max(ln.norm(mesh[:,list(np.arange(0,int(i+2),step = 2))]- pre,axis = 0))
    # #         # error = np.max(ln.norm(mesh[list(np.arange(0,int(i+2),step = 2)),:] - pre,axis = 1))
    # #         print(error)
    # #     pre = mesh

    for i in list(np.linspace(0,num,num = num + 1)):
        sk = start * np.power(s_pow,i)
        tn = start * np.power(t_pow,i)
        print('sk = ',sk,' ','tn = ',tn)
        mesh, boudary = AmerPut.scheme(sk,tn)
        AmerPut.scheme_plot(mesh, boudary,sk,tn)
        if pre !=[]:
            indexing_s = list(np.arange(0,int(sk + s_pow),step = s_pow))
            indexing_t = list(np.arange(0,int(tn + t_pow),step = t_pow))
            error = np.max(np.abs(mesh[np.ix_(indexing_s, indexing_t)] - pre))
            print(error)
        pre = mesh
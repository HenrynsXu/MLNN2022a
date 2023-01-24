import numpy as np
import math
import matplotlib.pyplot as plt
class Hopfield:
    def __init__(self, city_loc, cycles=1000, u0=0.2, dt=0.001, A=5,D=3):
        '''
        city_loc: 城市位置，用数组表示
        cycles：神经网络迭代次数
        u0：U矩阵计算基准值
        dt：步长，相当于学习率
        A，D为惩罚常数
        '''
        self.city_loc = city_loc
        self.cyc = cycles
        self.dt = dt
        self.A = A
        self.D = D
        self.city_num = len(city_loc)
        self.u0 = u0
        
    def distance(self): # 求城市间距离，表示成矩阵
        mat_d = np.zeros((self.city_num,self.city_num))
        for i in range(1,self.city_num):
            for j in range(i):
                mat_d[i,j] = math.sqrt((self.city_loc[i,0]-self.city_loc[j,0])**2+
                                        (self.city_loc[i,1]-self.city_loc[j,1])**2)
                mat_d[j,i] = mat_d[i,j]  # 对称矩阵
        return mat_d
    def init_v(self):
        seq = np.arange(self.city_num)
        np.random.shuffle(seq)
        v = np.zeros((self.city_num,self.city_num))
        for i in range(self.city_num):
            v[seq[i],i]=1
        return v
    def cal_du_dt(self,v,d):
        du_dt = np.zeros((self.city_num,self.city_num))
        for x in range(self.city_num):
            for i in range(self.city_num):
                sumdv=0
                sumvxj = v[x,:].sum()
                sumvyi = v[:,i].sum()
                for y in range(self.city_num):                
                    if i<self.city_num-1:
                        sumdv+=d[x,y]*v[y,i+1]
                    else:
                        sumdv+=d[x,y]*v[y,0]
                du_dt[x,i]=-self.A*(sumvxj-1)-self.A*(sumvyi-1)-self.D*sumdv
        return du_dt
    def update_u(self,ut,du_dt):
        return ut+du_dt*self.dt
    def update_v(self,u):    
        return (1+np.tanh(u/self.u0))/2
    def energy(self,v,d):
        temp1 = 0
        temp2 = 0
        temp3 = 0
        for x in range(self.city_num):
            temp1 += (v[x,:].sum()-1)**2
        for i in range(self.city_num):
            temp2 += (v[:,i].sum()-1)**2
        for x in range(self.city_num):
            for y in range(self.city_num):
                for i in range(self.city_num-1):
                    temp3 += v[x,i]*d[x,y]*v[x,i]*v[y,i+1]
                temp3 += v[x,self.city_num-1]*d[x,y]*v[x,self.city_num-1]*v[y,0]
        return self.A*temp1/2+self.A*temp2/2+self.D*temp3/2
    def hopmain(self):
        v0 = self.init_v()
        d = self.distance()
        e = []
        u0 = np.random.normal(size=(self.city_num,self.city_num))
        for iter in range(self.cyc):
            du_dt = self.cal_du_dt(v0,d)
            u = self.update_u(u0,du_dt)
            v = self.update_v(u0)
            u0 = u
            v0 = v
            e.append(self.energy(v,d))
        return v0,e
if __name__ == '__main__':
    city_loc = np.array([[1,1],[1,2],[2,2],[2,1]])
    hf = Hopfield(city_loc)
    re, e = hf.hopmain()
    print(re)
    times = hf.cyc
    plt.plot(range(times),e)
    plt.show()

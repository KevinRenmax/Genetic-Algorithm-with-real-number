import numpy as np
import numpy.random as npr
class Individual:
    _n=0
    eval=0.0
    chromsome=None
    def __init__(self,n):
        self._n=n
        self.chromsome=npr.random(n)

class NGA:
    population=[] #种群
    dimension=1
    bestPos=worstPos=0
    mutationProb=10 #变异概率
    crossoverProb=90 #交叉概率
    maxIterTime=1000 #最大迭代次数
    evalFunc=None
    arfa =1.0
    popu=2 #个体数量
    def __init__(self,popu, dimension,crossoverProb,mutationProb,maxIterTime,evalFunc):
        for i in range(popu):
            oneInd=Individual(dimension)
            oneInd.eval=evalFunc(oneInd.chromsome)
            self.population.append(oneInd)

        self.crossoverProb=crossoverProb
        self.mutationProb=mutationProb
        self.maxIterTime=maxIterTime
        self.evalFunc=evalFunc
        self.popu=popu
        self.dimension=dimension

    #找最好的个体位置
    def findBestWorst(self):
        worst=best= self.population[0].eval
        worstPos=bestPos = 0

        for i in range(1,self.popu):
            if best > self.population[i].eval:
                bestPos = i
                best = self.population[i].eval
            if worst<self.population[i].eval:
                worstPos=i
                worst=self.population[i].eval
        self.bestPos=bestPos
        self.worstPos=worstPos

       #交叉操作
    def crossover(self):
        fatherPos=npr.randint(0,self.popu)
        motherPos=npr.randint(0,self.popu)
        while motherPos == fatherPos:
            motherPos = npr.randint(0,self.popu)
        father = self.population[fatherPos]
        mother = self.population[motherPos]
        startPos = npr.randint(self.dimension) #交叉的起始位置
        jeneLength = npr.randint(self.dimension)+1 # //交叉的长度
        #jeneLength = self.dimension - startPos #  //基因交换的有效长度
        son1 = Individual(self.dimension)
        son2 = Individual(self.dimension)

        son1.chromsome[0:startPos]=father.chromsome[0:startPos]
        son2.chromsome[0:startPos]=mother.chromsome[0:startPos]

        left = startPos + jeneLength
        son1.chromsome[startPos:left]=mother.chromsome[startPos:left]
        son2.chromsome[startPos:left]=father.chromsome[startPos:left]


        son1.chromsome[left:]=father.chromsome[left:]
        son2.chromsome[left:]=mother.chromsome[left:]
        son1.eval = self.evalFunc(son1.chromsome) #;// 评估第一个子代
        son2.eval = self.evalFunc(son2.chromsome)
        self.findBestWorst()

        if son1.eval < self.population[self.worstPos].eval:
            self.population[self.worstPos] = son1
        self.findBestWorst()
        if son2.eval < self.population[self.worstPos].eval:
            self.population[self.worstPos] = son2

    def mutation(self):
        father = self.population[npr.randint(self.popu)]
        son = Individual(self.dimension)
        son.chromsome[0:]=father.chromsome[0:]
        mutationPos =npr.randint(self.dimension)#;//变异的位置
        #产生一个0-1之间的随机小数
        temp = npr.random()
        sign = npr.randint(0,2)   # ;//产生0 或1，决定+ 还是 -
        if sign == 0:
            temp = -temp
        son.chromsome[mutationPos]=father.chromsome[mutationPos] + self.arfa * temp
        son.eval = self.evalFunc(son.chromsome)
        self.findBestWorst()
        if son.eval < self.population[self.worstPos].eval:
            self.population[self.worstPos] = son

    def solve(self):
        shrinkTimes = self.maxIterTime / 10
        #//将总迭代代数分成10份
        oneFold = shrinkTimes #;//每份中包含的次数
        i = 0
        while i < self.maxIterTime:
            print(i,"---",self.maxIterTime)
            if i == shrinkTimes:
                self.arfa =self.arfa / 2.0
            #经过一份代数的迭代后，将收敛参数arfa缩小为原来的1/2，以控制mutation
            shrinkTimes += oneFold  #;//下一份到达的位置
            for j in range(self.crossoverProb):
                self.crossover()
            for j in range(self.mutationProb):
                self.mutation()
            print("solution:",self.population[self.bestPos].chromsome)
            print("func value:",self.population[self.bestPos].eval)
            i=i+1



    def getAnswer(self):
        self.findBestWorst()
        return self.population[self.bestPos].chromsome

import math
# def f2(v):
#     pred = [v[0] * v[1] * (math.exp(-v[1] * i) - math.exp(-v[2] * i)) / (v[2] - v[1]) for i in x]
#     error=y-pred
#     s=np.sum(error*error)
#     return s

# import xlrd
# x = xlrd.open_workbook(r'E:\CS\机器学习\第一次大作业 二元线性回归/归一化后.xlsx')
# y = x.sheets()[0]
# lk = []
# for i in range(1, y.nrows-6):
#     row = y.row_values(i)
#     aa = [float(row[2]),float(row[3])]
#     lk.append(aa)
#
# lktest = []
# for i in range(15, y.nrows):
#     row = y.row_values(i)
#     aa = [float(row[2]),float(row[3])]
#     lktest.append(aa)
#
# y1 = []
# for i in range(1, y.nrows-6):
#     row = y.row_values(i)
#     aa = float(row[1])
#     y1.append(aa)
#
# x_train = np.array(lk)
# y_train = np.array(y1)
# x_test = np.array(lktest)
#
# def f3(v):
#     pred = [v[0]*i[0] + v[1]*i[1] + v[2] for i in x_train]
#     error = y_train - pred
#     s = np.sum(error**2)
#     return s


# ma = np.loadtxt(r'E:\CS\python\第三讲 初等建模/花青素.txt')
# x=ma[:,0]
# y=ma[:,1]

#
# nga=NGA(50,3,50,150,5000,f3)
# nga.solve()
# ans=nga.getAnswer()
# print(ans)







# def f1(v):
#     f = (v[0] - 4)**2 + 2 * (v[1] + 3)** 2+ (v[2] - 4)** 2
#     return f
# #
# import math
# def f2(v): # 黑枸杞函数
#     pred=[v[0] * v[1] * (math.exp(-v[1] * i) - math.exp(-v[2] * i)) / (v[2] - v[1]) for i in x ]
#     error=y-pred
#     s=sum(error*error)
#     return s
    
    
   

# ma = np.loadtxt(r"G:\teach\2017sitp\前400min.txt")
# x=ma[:,0]
# y=ma[:,1]
#
# def f3(v):
#     sum = math.fabs(2*v[1]+3) +math.fabs(4*v[0]*v[0]+math.sin(v[1]*v[2])) +math.fabs(v[1]*v[2]/2-3)
#     return sum
#
#
#
#
#
# nga=NGA(10,3,0,90,1000,f2)
# nga.solve()
# ans=nga.getAnswer()
# print(ans)



# nga = NGA(20,3,30,70,100,f1)
# nga.solve()
# ans = nga.getAnswer()
# print(ans)

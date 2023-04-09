import random
import numpy as np
def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def NeuralNetwork(x, y, w, b, rate, m ,k):
    for j in range(k):
        J=0
        db1 = np.array([[0.,0.]])
        db2 = np.array([[0.]])
        db=[db1,db2]
        
        dw1 = np.array([[0.,0.],[0.,0.]])
        dw2 = np.array([[0.,0.]])
        
        dw=[dw1,dw2]
        
        z1 = w[0]@x+b[0]
        a1 = Sigmoid(z1)
        
        z2 = w[1]@a1 + b[1]
        a2 = Sigmoid(z2)
        
        J = np.sum(-(y*np.log(a2)+(1-y)*np.log(1-a2)))
        
        dz2 = a2 - y
        dw[1] = dz2@a1.T
        db[1] = np.sum(dz2,axis=1,keepdims=1)
        da1 = w[1].T@dz2
        dz1 = da1*a1*(1-a1)
        
        dw[0] = dz1@x.T
        db[0] = np.sum(dz1,axis=1,keepdims=1)
        
        J /= m
        dw[0] /= m
        dw[1] /= m
        
        db[0] /= m
        db[1] /= m
        
        w[0] -= rate*dw[0]
        w[1] -= rate*dw[1]
        b[0] -= rate*db[0]
        b[1] -= rate*db[1]
        
        if j%500 ==499:
            print(j+1, J)
    return w, b

def Score(x, y, w, b, n):
    score = 0
    z1 = w[0]@x+b[0]
    a1 = Sigmoid(z1)
    
    z2 = w[1]@a1 + b[1]
    a2 = Sigmoid(z2)
    for i in range(n):
        if a2[0][i]>0.5 and y[0][i]==1:
            score+=1
        elif a2[0][i]<=0.5 and y[0][i]==0: 
            score+=1
    score_ratio = score/n*100
    return score, score_ratio


m_list=[10, 100, 10_000, 10_000, 10_000, 10_000]
k_list = [5000, 5000, 5000, 10, 100, 5000]
# lr = [0.961, 0.507, 0.996, 0.92 ,0.89, 0.998]
lr = [9.4, 0.918, 0.999, 0.94, 0.919, 0.999]
scores = [(0.,0.) for _ in range(len(m_list))] 
# lr = [0. for _ in range(len(m_list))]
rates = [rate/1000 for rate in range(900,1001)]
estimated = [(0.,0.) for _ in range(len(m_list))] 
n = 1000


# for idx, rate in enumerate(rates):
for idx, rate in enumerate(lr):
        l = idx
    # for l in range(len(m_list)):
        m = m_list[l]
        k = k_list[l]
        x1_train=[]
        x2_train=[]
        y_train=[]

        x1_test=[]
        x2_test=[]
        y_test=[]
        
        for i in range(m):
            x1_train.append(random.uniform(-10,10))
            x2_train.append(random.uniform(-10,10))
            if x1_train[-1] <-5 or x1_train[-1] > 5:
                y_train.append(1)
            else:   
                y_train.append(0)
        for i in range(n):
            
            x1_test.append(random.uniform(-10,10))
            x2_test.append(random.uniform(-10,10))
            if x1_test[-1] <-5 or x1_test[-1] > 5:
                y_test.append(1)
            else:   
                y_test.append(0)

        weight1 = np.array([[-2.0, 1.0], [2.0, -1.0]])
        weight2 = np.array([[1.0,-2.0]])
        weight = [weight1,weight2]
        b1 = np.array([[-5.,5.]]).reshape(2,1)
        b2 = np.array([[1. ]])
        b = [b1,b2]
        x = np.row_stack((x1_train,x2_train))
        x_test = np.row_stack((x1_test,x2_test))
        y = np.array(y_train)[np.newaxis,:]
        y_test = np.array(y_test)[np.newaxis,:]
        weight, b = NeuralNetwork(x, y, weight, b, rate, m, k)
        score_train, score_ratio_train = Score(x, y, weight, b, m)
        score_test, score_ratio_test = Score(x_test,y_test,weight,b,n)
        
        print("idx = {0}".format(idx), end= ' ')
        print("m = {0}, n = {1}, K = {2}".format(m,n,k))
        print("Accuracy of Train = {0}".format(score_ratio_train))
        print("Accuracy of Test = {0}".format(score_ratio_test))
        
        if score_ratio_train >= scores[l][0]:
            if score_ratio_test >= scores[l][1]:
                scores[l]=(score_ratio_train,score_ratio_test)
                lr[l]=rate
                estimated[l] = (weight,b)
        elif score_ratio_train >= 98.0:
            if score_ratio_test >= scores[l][1]:
                scores[l]=(score_ratio_train,score_ratio_test)
                lr[l]=rate
                estimated[l] = (weight,b)
                
            # else:
            #     delta1 = (score_ratio_train - score_ratio_test)**2
            #     delta2 = (scores[l][0] - scores[l][1])**2
            #     if delta1<delta2:
            #         scores[l]=(score_ratio_train,score_ratio_test)
            #         lr[l]=rate
print("(score_ratio_train, score_ratio_test) = ", scores)
print(lr)
for i in range(len(m_list)):
    print("weight : ", estimated[i][0],"b : ", estimated[i][1])

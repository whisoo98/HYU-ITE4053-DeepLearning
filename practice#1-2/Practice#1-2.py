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
            print(Score(x,y,w,b,m))
    return w, b

def Score(x, y, w, b, m):
    score = 0
    z1 = w[0]@x+b[0]
    a1 = Sigmoid(z1)
    
    z2 = w[1]@a1 + b[1]
    a2 = Sigmoid(z2)
    print(a2)
    for i in range(m):
        if a2[0][i]>0.5 and y[0][i]==1:
            score+=1
        elif a2[0][i]<=0.5 and y[0][i]==0:
            score+=1
    score_ratio = score/m*100
    return score, score_ratio
x1_train=[]
x2_train=[]
y_train=[]

rate = 1
m=10_000
k = 5000

for i in range(m):
    x1_train.append(random.uniform(-10,10))
    x2_train.append(random.uniform(-10,10))
    if x1_train[-1] <-5 or x1_train[-1] > 5:
        y_train.append(1)
    else:   
        y_train.append(0)

weight1 = np.array([[-2.0, 1.0], [2.0, -1.0]])
weight2 = np.array([[1.0,-2.0]])
weight = [weight1,weight2]
b1 = np.array([[-5.,5.]]).reshape(2,1)
b2 = np.array([[1. ]])
b = [b1,b2]
x = np.row_stack((x1_train,x2_train))
y = np.array(y_train)[np.newaxis,:]
weight, b = NeuralNetwork(x, y, weight, b, rate, m, k)
score, score_ratio = Score(x, y, weight, b, m)
print(score, score_ratio)
print(weight, b)
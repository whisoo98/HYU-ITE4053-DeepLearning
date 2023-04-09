import random
import numpy as np
def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

# def LogisticRegression(x,y,weight, b, rate, m ,k):
#     for j in range(k):
#         J=0
#         dw=np.zeros(2)
#         db=0
#         for i in range(m):
#             z = np.dot(weight,x[i])+b
#             a = Sigmoid(z)
#             J += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
#             dz = a-y[i]
#             dw += x[i]*dz
#             db += dz
#         J /= m
#         dw /= m
#         db /= m
#         weight -= rate*dw
#         b -= rate*db
#         if j%1000==999:
#             print(j)
#     return weight, b
def LogisticRegression(x1,x2,y,weight, b, rate, m ,k):
    for j in range(k):
        J=0
        dw1 = 0
        dw2 = 0
        db=0
        for i in range(m):
            z1 = weight[0]*x1[i]
            z2 = weight[1]*x2[i]
            z = z1 + z2 + b
            a = Sigmoid(z)
            J += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            dz = a-y[i]
            dw1 += x1[i]*dz
            dw2 += x2[i]*dz
            db += dz
        J /= m
        dw1 /= m
        dw2 /= m
        db /= m
        weight[0] -= rate*dw1
        weight[1] -= rate*dw2
        b -= rate*db
        if j%1000==999:
            print(j)
    return weight, b
# def Score(x, y, weight, b, m):
#     score = 0
#     for i in range(m):
#         z = np.dot(weight,x[i])+b
#         a = Sigmoid(z)
#         if a==y[i]:
#             score+=1
#     score_ratio = score/m*100
#     return score, score_ratio
def Score(x1,x2, y, weight, b, m):
    score = 0
    for i in range(m):
        z1 = weight[0]*x1[i]
        z2 = weight[1]*x2[i]
        z = z1 + z2 + b
        a = Sigmoid(z)
        if a>0.5 and y[i]==1:
            score+=1
        elif a<=0.5 and y[i]==0:
            score+=1
    score_ratio = score/m*100
    return score, score_ratio
x1_train=[]
x2_train=[]
y_train=[]
weight = [-5,2]
#weight = np.array([-5.0,2.0])
#x = np.column_stack((x1_train,x2_train))
b = 10
rate = 0.01
m=10_000
k = 5000

for i in range(m):
    x1_train.append(random.uniform(-10,10))
    x2_train.append(random.uniform(-10,10))
    if x1_train[-1] <-5 or x1_train[-1] > 5:
        y_train.append(1)
    else:   
        y_train.append(0)

# weight, b = LogisticRegression(x,y_train,weight, b,rate,m,k)
# score, score_ratio = Score(x, y_train,weight,b,m)
weight, b = LogisticRegression(x1_train, x2_train,y_train,weight, b,rate,m,k)
score, score_ratio = Score(x1_train, x2_train, y_train,weight,b,m)
print(score, score_ratio)
print(weight, b)
import numpy as np

W, b = np.array([0, 0]), 0.0
learning_rate = 0.01 

def activation(s):
    if s >= 0: return 1
    else: return 0

def out(x) :
    return activation (W.dot(x) + b)

def train(x0, x1, target):
    global W, b
    X = np.array([x0, x1])
    y = out(X)

    ### 예측이 맞으면 아무것도 하지 않음-------------------------------------
    if target == y: return False         # 가중치가 변경되지 않았음을 반환
    ### 예측이 틀리면 학습 실시---------------------------------------------
    print('가중치 수정전 X : {} target :{} y:{} b:{} W:{}'.format(X,target, y, b, W))
    W = W - learning_rate * X * (y - target)   # 오차에 비례하여 가중치 변경
    b = b - learning_rate * 1 * (y - target)   # 편향: 입력이 1이라고 볼 수 있음
    print('가중치 수정후 X : {} target :{} y:{} b:{} W:{}'.format(X,target, y, b, W))
    return True 

adjusted = 0
for i in range(100):
    adjusted += train(0, 0, 0)    # 훈련 데이터 1
    adjusted += train(0, 1, 0)    # 훈련 데이터 2
    adjusted += train(1, 0, 0)    # 훈련 데이터 3
    adjusted += train(1, 1, 1)    # 훈련 데이터 4
    print("iteration -------------", i)
    if not adjusted: break  # 모든 훈련에 대해 가중치 변화 없으면 학습종료
    adjusted = 0

def predict(inputs):
    outputs = []
    for x in inputs:
        outputs.append (out(x))
    return outputs
    
X = [[0, 0], [0, 1], [1, 0], [1,1]]
yhat = predict(X)
print('x0 x1  y')
for i in range(len(X)):
    print('{0:2d} {1:2d} {2:2d}'.format(X[i][0], X[i][1], yhat[i]))
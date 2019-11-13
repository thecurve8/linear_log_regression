import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from functools import wraps
from sklearn.utils import shuffle

"""
Created on Tue Jan 22 15:36:08 2019

@author: Alexander APOSTOLOV (1005644279) and Anthony KEMMEUGNE (1004686789)
"""

#Load the dataPoints and labels of the Training, Validation and Test sets
def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
        #print(trainData)
    return trainData, validData, testData, trainTarget, validTarget, testTarget

######################################
##---------------MSE----------------##
######################################
    
#Calculates the MSE of the dataset x with correct labels y using
#weights W and bias b and regularization parameter reg
#W: weights, b: bias, x:datapoints, y:labels, reg:regularization parameter
def MSE(W, b, x, y, reg):
    y_hat = np.add(np.matmul(x,W),b)
    error = np.subtract(y_hat,y)
    squared = np.square(error)
    LD = np.sum(squared)
    LD = LD/(2*N)
    pen_term = reg*np.sum(np.square(W))
    LW= pen_term/2
    L = LW+LD
    return L

#Calculates the gradient of the MSE loss function according
#to all the weights and the bias
#W: weights, b: bias, x:datapoints, y:labels, reg:regularization parameter
def gradMSE(W, b, x, y, reg):
    pre_prediction = np.matmul(x, W)
    prediction = np.add(pre_prediction, b)
    e = np.subtract(prediction,y)
    regTerm = np.multiply(W, reg)
    tempor = np.matmul(np.transpose(x), e)
    scaled = np.multiply(tempor, 1/N)
    gradW = np.add(scaled, regTerm)
    gradb=(np.sum(e))/N
    return gradW, gradb

#Calculates the MSE and accuracy of the dataset x with correct labels
#y using weights W and bias b and regularization parameter reg
#Accuracy is calculated as the ratio of correctly predicted datapoints
#using threshold of 0.5 
#W: weights, b: bias, x:datapoints, y:labels, reg:regularization parameter, size: size of the dataset
def statistics(W, b, x, y, reg, size):
    y_hat = np.add(np.matmul(x,W),b)
    error = np.subtract(y_hat,y)
    squared = np.square(error)
    LD = np.sum(squared)
    LD = LD/(2*size)
    pen_term = reg*np.sum(np.square(W))
    LW= pen_term/2
    L = LW+LD
    accuracy = 1 - np.sum(abs(np.subtract(np.around(np.clip(y_hat, 0, 1)),y)))/size
    return L, accuracy



######################################
##---------------CE-----------------##
######################################
    
#Calculates the CE loss of the dataset x with correct labels y using
#weights W and bias b and regularization parameter reg
#W: weights, b: bias, x:datapoints, y:labels, reg:regularization parameter
def crossEntropyLoss(W, b, x, y, reg):
    xw = np.add(np.matmul(x,W),b)
    mul = np.multiply(np.subtract(1,y),xw)
    nxw = -1*xw
    val = np.add(1,np.exp(nxw))
    sect = np.log(val)
    LD = np.add(mul,sect)
    sLD = np.sum(LD)
    sLD = sLD/(N)
    pen_term = reg*np.sum(np.square(W))
    LW= pen_term/2
    L = LW+sLD
    return L
    

#Calculates the gradient of the CE loss function according
#to all the weights and the bias
#W: weights, b: bias, x:datapoints, y:labels, reg:regularization parameter
def gradCE(W, b, x, y, reg):
    lexp = np.matmul(x,W)
    e = np.add(lexp,b)
    ne=-e
    expo = np.exp(ne)
    ratio = expo/(1+expo)
    mratio = -1*ratio
    my = -1*y
    somw= np.add(1, np.add(mratio,my))
    regTerm = np.multiply(W, reg)
    tempor = np.matmul(np.transpose(x),somw)
    gradCEW = np.add((1/N)*tempor,regTerm)
    somb =  somw
    gradCEb = np.sum(somb)/N
    
    return gradCEW, gradCEb

#Calculates the CE loss and accuracy of the dataset x with correct labels
#y using weights W and bias b and regularization parameter reg
#Accuracy is calculated as the ratio of correctly predicted datapoints
#using threshold of 0.5 
#W: weights, b: bias, x:datapoints, y:labels, reg:regularization parameter, size: size of the dataset    
def statistics_CE(W, b, x, y, reg, size):
    y_hat =  1/(1+np.exp(-(np.matmul(x,W) + b)))
    xw = np.add(np.matmul(x,W),b)
    mul = np.multiply(np.subtract(1,y),xw)
    nxw = -1*xw
    val = np.add(1,np.exp(nxw))
    sect = np.log(val)
    LD = np.add(mul,sect)
    sLD = np.sum(LD)
    sLD = sLD/(size)
    pen_term = reg*np.sum(np.square(W))
    LW= pen_term/2
    L = LW+sLD
    accuracy = 1 - np.sum(abs(np.subtract(np.around(y_hat),y)))/size
    return L, accuracy
   
##############################################################################    
##############################################################################

#Gradient descent algorithm computing
#W: initial weights, b: initial bias, trainingData : trainingData set,
#trainingLabels: correct labels, alpha: alpha parameter, iterations: number of epochs,
#reg : regularization parameter, EPS: step at which to stop computations, losstype: "MSE" or "CE"
def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, losstype = "None"):

    CW = W
    Cb = b
    tot_step = 1
    it = 0

    start_time=time.time()
    prev_time=start_time
    cur_time=start_time
    
    if(losstype == "MSE"):
        print("Weights using batch GD with MSE loss function computed...")
    elif(losstype=="CE"):
        print("Weights using batch GD with CE loss function computed...")
    else:
        print("No loss function selected, please select MSE for mean square error or CE for binary cross entropy loss.")
        return None, None
    
    while ((tot_step > EPS) and (it < iterations)):
        prev_W = CW
        prev_b = Cb
        gW=0
        gB=0
        if(losstype == "MSE"):
            gW,gB = gradMSE(CW,Cb,trainingData, trainingLabels,reg)
        elif(losstype=="CE"):
            gW,gB = gradCE(CW,Cb,trainingData, trainingLabels,reg)
        CW = CW-alpha*gW
        Cb = Cb-alpha*gB
        step_size_W = np.sum(abs(np.subtract(CW, prev_W))) 
        step_size_b = abs(Cb-prev_b)
        prev_step=tot_step
        tot_step = step_size_W + step_size_b 

        ###INFORMATION SHOWN WHILE COMPUTING ###
        cur_time=time.time()
        if(it%500==0):
            print("At iteration ",it+1," The total step is ",tot_step)
            if(it >0 ):
                print("decrease of the error is ", (prev_step-tot_step))
            print("Last iteration done in ", (cur_time-prev_time), " total elapsed: ", (cur_time-start_time))
            loss=0
            if(losstype == "MSE"):
                loss = MSE(CW,Cb,trainingData, trainingLabels,reg)
            elif(losstype=="CE"):
                loss = crossEntropyLoss(CW,Cb,trainingData, trainingLabels,reg)
            print("The loss is: ", loss, "\n")
        ### --------------------------------- ###
        prev_time=cur_time
        
        if(takeStatsBatch):
            if(losstype == "MSE"):
                trainLoss, trainAcc = statistics(CW, Cb, trainingData, trainingLabels, reg, N)
                validLoss, validAcc = statistics(CW, Cb, flat_valid, validTarget, reg, N_valid)
                testLoss, testAcc = statistics(CW, Cb, flat_test, testTarget, reg, N_test)               
            elif(losstype == "CE"):    
                trainLoss, trainAcc = statistics_CE(CW, Cb, trainingData, trainingLabels, reg, N)
                validLoss, validAcc = statistics_CE(CW, Cb, flat_valid, validTarget, reg, N_valid)
                testLoss, testAcc = statistics_CE(CW, Cb, flat_test, testTarget, reg, N_test)
            lossHistory_reg[it]=(trainLoss, validLoss, testLoss)
            accuracyHistory_reg[it]=(trainAcc, validAcc, testAcc)
            
        it = it + 1
    return CW, Cb

#Computes the wights with the normal equation
#x: datapoints, y:correct labels
def closed_form(x,y):
    inner = np.matmul(np.transpose(x),x)
    inversed = np.linalg.inv(inner)
    res = np.matmul(np.matmul(inversed, np.transpose(x)), y)
    return res
   
#Build graph for the tenserflow minibatch algorithm using ADAM
#loss: loss type "MSE" or "CE"
def buildGraph(loss=None):
    tf.set_random_seed(421)
    w = tf.Variable(tf.truncated_normal((784,1), 0, 0.5, tf.float32,None, None))
    b = tf.Variable(0.0, name = 'biases')
    x = tf.placeholder(tf.float32, shape=(None, 784))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    reg = tf.placeholder(tf.float32)
    tf.set_random_seed(421)
    if loss == "MSE":
        y_hat = tf.matmul(x, w) + b
        loss = tf.reduce_mean(tf.reduce_mean(tf.square(y_hat-y)))

    elif loss == "CE":
        y_hat = 1/(1+tf.exp(-(tf.matmul(x,w) + b)))
        epsilon = 8e-8
        log1 = tf.log(y_hat + epsilon)
        log2 = tf.log(-y_hat + 1 + epsilon)
        loss = tf.reduce_mean( tf.multiply(-y,log1) - tf.multiply(1-y,log2) )
        
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
    return w, b, y_hat, y, loss, train_op, reg, x


#SGD algorithm using ADAM
#batchSize: size of the minibatches, epochs: number of epochs to run, lossType: "MSE" or "CE"
def stochastichGD(batchSize, epochs, lossType = "None"):
    w, b, y_hat, y, loss, optimizer, reg, x = buildGraph(lossType)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(0, epochs):
            flat_x_shuffled,trainingLabels_shuffled = shuffle(flat_x, trainingLabels)
            number_of_batches = N//batchSize

            for minibatch_index in range(0,number_of_batches):
                #select miniatch and run optimizer
                minibatch_x = flat_x_shuffled[minibatch_index*batchSize: (minibatch_index + 1)*batchSize, :]
                minibatch_y = trainingLabels_shuffled[minibatch_index*batchSize: (minibatch_index + 1)*batchSize]
                _, new_w, new_b, new_loss, new_y_hat =  sess.run([optimizer, w, b, loss, y_hat], feed_dict = {x : minibatch_x, y: minibatch_y, reg:0.0})
            
            if(takeStatsAdam):
                if lossType == "MSE":
                    loss_after_batch_train, accu_after_batch_train = statistics(new_w, new_b, flat_x, trainingLabels, 0, N)
                    loss_after_batch_valid, accu_after_batch_valid = statistics(new_w, new_b, flat_valid, validTarget, 0, N_valid)
                    loss_after_batch_test, accu_after_batch_test = statistics(new_w, new_b, flat_test, testTarget, 0, N_test)
                    loss_history_adam[i] = (loss_after_batch_train,loss_after_batch_valid, loss_after_batch_test)
                    accuracy_history_adam[i] = (accu_after_batch_train, accu_after_batch_valid, accu_after_batch_test)
                elif lossType == "CE":
                    loss_after_batch_train, accu_after_batch_train = statistics_CE(new_w, new_b, flat_x, trainingLabels, 0, N)
                    loss_after_batch_valid, accu_after_batch_valid = statistics_CE(new_w, new_b, flat_valid, validTarget, 0, N_valid)
                    loss_after_batch_test, accu_after_batch_test = statistics_CE(new_w, new_b, flat_test, testTarget, 0, N_test)
                    loss_history_adam[i] = (loss_after_batch_train,loss_after_batch_valid, loss_after_batch_test)
                    accuracy_history_adam[i] = (accu_after_batch_train, accu_after_batch_valid, accu_after_batch_test) 

            if(giveFinalPerformanceAdam):
                if(i==epochs-1):
                    if(lossType == "MSE"):
                        loss_after_batch_train, accu_after_batch_train = statistics(new_w, new_b, flat_x, trainingLabels, 0, N)
                        loss_after_batch_valid, accu_after_batch_valid = statistics(new_w, new_b, flat_valid, validTarget, 0, N_valid)
                        loss_after_batch_test, accu_after_batch_test = statistics(new_w, new_b, flat_test, testTarget, 0, N_test)
                        print("Train")
                        print(accu_after_batch_train)
                        print(loss_after_batch_train)                    
                        print("Valid")
                        print(accu_after_batch_valid)
                        print(loss_after_batch_valid)
                        print("Test:")
                        print(accu_after_batch_test)
                        print(loss_after_batch_test)                    
                    elif(lossType == "CE"):
                        loss_after_batch_train, accu_after_batch_train = statistics_CE(new_w, new_b, flat_x, trainingLabels, 0, N)
                        loss_after_batch_valid, accu_after_batch_valid = statistics_CE(new_w, new_b, flat_valid, validTarget, 0, N_valid)
                        loss_after_batch_test, accu_after_batch_test = statistics_CE(new_w, new_b, flat_test, testTarget, 0, N_test)
                        print("Train")
                        print(accu_after_batch_train)
                        print(loss_after_batch_train)                    
                        print("Valid")
                        print(accu_after_batch_valid)
                        print(loss_after_batch_valid)
                        print("Test:")
                        print(accu_after_batch_test)
                        print(loss_after_batch_test) 
    return


################################################################
##    END OF FUNCTIONS                                        ##
################################################################


##----------------Global Variables----------------------------##
N=3500
N_valid =100
N_test = 145
dimw = 784

takeStatsAdam = False
takeStatsBatch = False
giveFinalPerformanceAdam = False

trainingData, validData, testData, trainingLabels, validTarget, testTarget = loadData();
#We flatten all the dataSets to use them in our functions, images are know [784, 1] instead of [28, 28]
flat_x = np.zeros((N,dimw))
for i in range(0,N):
    flat_x[i]=trainingData[i].flatten()
flat_valid = np.zeros((N_valid, dimw))
for i in range(0,N_valid):
    flat_valid[i]=validData[i].flatten()
flat_test = np.zeros((N_test, dimw))
for i in range(0,N_test):
    flat_test[i]=testData[i].flatten()

epochs=5000

##TO USE WHEN PLOTS ARE NEEDED
if(takeStatsAdam):
    loss_history_adam = np.zeros((epochs,3,1))
    accuracy_history_adam = np.zeros((epochs,3,1))

if(takeStatsBatch):
    lossHistory_reg=np.zeros((epochs,3,3))
    accuracyHistory_reg=np.zeros((epochs,3,3))




##################################################################
##################################################################


CW = np.zeros((dimw,1))
Cb = 0
EPS = 1e-7
alpha = 0.005
reg=0.1

###objsRegular can be used to save the histories

##Use this to load the histories
#with open('objsRegular.pkl', 'rb') as f:  
#    accuracyHistory_reg, lossHistory_reg = pickle.load(f)

#start=time.time()
#CW, Cb = grad_descent(CW, Cb, flat_x, trainingLabels, alpha, epochs, reg, EPS, "CE")
#end = time.time()
#print("Computation of weights ended in :", end-start)

##Use this to save the histories
#with open('objsRegular.pkl', 'wb') as f: 
#    pickle.dump([accuracyHistory_reg, lossHistory_reg], f)
   
##-------------SGD-ADAMA------------------##
#with open('objsAdam.pkl', 'rb') as f:  
#    accuracy_history_adam, loss_history_adam = pickle.load(f)

#start=time.time()
#stochastichGD(100, 5000, "CE")
#end=time.time()
#print(end-start)

#with open('objsAdam.pkl', 'wb') as f: 
#    pickle.dump([accuracy_history_adam, loss_history_adam], f)



 
################################################################
##EVERYTHING BELOW IS USED TO PLOT DATA IN DIFFERENT SCENARIOS##
################################################################

#plt.title("Loss with CE")
#plt.plot(lossHistory_reg[:,0,0], '-b', label='TrainData')
#plt.plot(lossHistory_reg[:,0,1], '-r', label='ValidationData')
#plt.plot(lossHistory_reg[:,0,2], '-g', label='TestData')
#plt.legend(loc='best')
#plt.xlabel('Epochs')
#plt.show()
#
#plt.title("Accuracy with CE")
#plt.plot(accuracyHistory_reg[:,0,0], '-b', label='TrainData')
#plt.plot(accuracyHistory_reg[:,0,1], '-r', label='ValidationData')
#plt.plot(accuracyHistory_reg[:,0,2], '-g', label='TestData')
#plt.legend(loc='best')
#plt.xlabel('Epochs')
#plt.show()


#plt.title("Accuracy using SGD ADAM and MSE and batch size of 100")
#plt.plot(accuracy_history_adam[:,0], '-b', label='TrainData')
#plt.plot(accuracy_history_adam[:,0], '-r', label='ValidationData')
#plt.plot(accuracy_history_adam[:,0], '-g', label='TestData')
#plt.legend(loc='best')
#plt.xlabel('Epochs')
#plt.show()
##
#plt.title("Loss using SGD ADAM and MSE and batch size of 100")
#plt.plot(loss_history_adam[:,0], '-b', label='TrainData')
#plt.plot(loss_history_adam[:,0], '-r', label='ValidationData')
#plt.plot(loss_history_adam[:,0], '-g', label='TestData')
#plt.legend(loc='best')
#plt.xlabel('Epochs')
#plt.show()
#
#for i in range(0, 5000):
#    if(np.isnan(accuracy_history_adam[i,0])):
#        accuracy_history_adam[i,0]=accuracy_history_adam[i-1,0]
#    if(np.isnan(loss_history_adam[i,0])):
#        loss_history_adam[i,0]=0.0005
#
#plt.title("Comparing Accuracy SGD-ADAM with batch gradient descent - CE")
#plt.plot(accuracy_history_adam[:,0], '-b', label='SGD-ADAM')
#plt.plot(accuracyHistory_reg[:,0], '-r', label='Batch')
#plt.legend(loc='best')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.show()
#
#plt.title("Comparing Loss SGD-ADAM with batch gradient descent - CE")
#plt.plot(loss_history_adam[:,0], '-b', label='SGD-ADAM')
#plt.plot(lossHistory_reg[:,0], '-r', label='Batch')
#plt.ylim(0,1)
#plt.legend(loc='best')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.show()
#
#plt.title("Comparing Accuracy SGD-ADAM with batch gradient descent - CE")
#x = range(2000,5000)
#plt.plot(x,accuracy_history_adam[2000:,0], '-b', label='SGD-ADAM')
#plt.plot(x,accuracyHistory_reg[2000:,0], '-r', label='Batch')
#plt.legend(loc='best')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.show()
#
#plt.title("Comparing Loss SGD-ADAM with batch gradient descent - CE")
#x = range(2000,5000)
#plt.plot(x,loss_history_adam[2000:,0], '-b', label='SGD-ADAM')
#plt.plot(x,lossHistory_reg[2000:,0], '-r', label='Batch')
#plt.legend(loc='best')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.show()


    
#start = time.time()  
#CW_closed = closed_form(flat_x, trainingLabels)
#end=time.time()
#print("Close formula finished in :", end-start)
#loss_closed, accu_closed = statistics(CW_closed, 0, flat_test, testTarget, 0, N_test)
#print("Loss: ", loss_closed, " Accuracy : ", accu_closed)

#for i in range(0,20):
#    print("Image: ", i ,": \n")
#    plt.imshow(validData[i], cmap="gray")
#    plt.show()
#    prediction=np.matmul(np.transpose(CW), flat_valid[i])+ Cb
#    print("Prediction : ", prediction)
#    print("Real value : ", validTarget[i])
#


def plot_for_given_alpha(data, beg, end, alphaN, dataName, increasing):
    alphaValue = "0.005"
    if(alphaN==1):
        alphaValue="0.001"
    if(alphaN==2):
        alphaValue="0.0001"
    plt.title(dataName + " of training, validation and test data for lambda=0 and alpha="+ alphaValue)
    x = range(beg, end)
    plt.plot(x,data[beg:end,alphaN,0], '-b', label='TrainData')
    plt.plot(x,data[beg:end,alphaN,1], '-r', label='ValidationData')
    plt.plot(x,data[beg:end,alphaN,2], '-g', label='TestData')
    plt.legend(loc='best')
    plt.ylabel(dataName)
    plt.xlabel('Epochs')
    if(increasing):
        minValue = min(data[beg,alphaN,:])
        maxValue = max(data[end-1,alphaN,:]) 
    else:
        minValue = min(data[end-1,alphaN,:])
        maxValue = max(data[beg,alphaN,:]) 
    interval = maxValue-minValue
    plt.ylim(minValue-interval*0.2, maxValue+interval*0.2)
    plt.show()
    return

def plot_alpha_comparison(data, beg, end, dataType, dataName, increasing):
    dataValue = "training"
    if(dataType==1):
        dataValue="validation"
    if(dataType==2):
        dataValue="test"
    plt.title(dataName+" comparison of "+ dataValue +" data for different values of alpha with lambda=0")
    x = range(beg, end)
    plt.plot(x,data[beg:end,0,dataType], '-b', label='alpha=0.005')
    plt.plot(x,data[beg:end,1,dataType], '-r', label='alpha=0.001')
    plt.plot(x,data[beg:end,2,dataType], '-g', label='alpha=0.0001')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    if(increasing):
        minValue = min(data[beg,:,dataType])
        maxValue = max(data[end-1,:,dataType]) 
    else:
        minValue = min(data[end-1,:,dataType])
        maxValue = max(data[beg,:,dataType]) 
    interval = maxValue-minValue
    plt.ylim(minValue-interval*0.2, maxValue+interval*0.2)
    plt.show()
    return 


def plot_for_given_reg(data, beg, end, regN, dataName, increasing):
    regValue = "0.001"
    if(regN==1):
        regValue="0.1"
    if(regN==2):
        regValue="0.5"
    plt.title(dataName + " of training, validation and test data for lambda="+regValue+" and alpha=0.005")
    x = range(beg, end)
    plt.plot(x,data[beg:end,regN,0], '-b', label='TrainData')
    plt.plot(x,data[beg:end,regN,1], '-r', label='ValidationData')
    plt.plot(x,data[beg:end,regN,2], '-g', label='TestData')
    plt.legend(loc='best')
    plt.ylabel(dataName)
    plt.xlabel('Epochs')
    if(increasing):
        minValue = min(data[beg,regN,:])
        maxValue = max(data[end-1,regN,:]) 
    else:
        minValue = min(data[end-1,regN,:])
        maxValue = max(data[beg,regN,:]) 
    interval = maxValue-minValue
    plt.ylim(minValue-interval*0.2, maxValue+interval*0.2)
    plt.show()
    return

def plot_reg_comparison(data, beg, end, dataType, dataName, increasing):
    dataValue = "training"
    if(dataType==1):
        dataValue="validation"
    if(dataType==2):
        dataValue="test"
    plt.title(dataName+" comparison of "+ dataValue +" data for different values of lambda with alpha=0.005")
    x = range(beg, end)
    plt.plot(x,data[beg:end,0,dataType], '-b', label='lambda=0.001')
    plt.plot(x,data[beg:end,1,dataType], '-r', label='lambda=0.1')
    plt.plot(x,data[beg:end,2,dataType], '-g', label='lambda=0.5')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    if(increasing):
        minValue = min(data[beg,:,dataType])
        maxValue = max(data[end-1,:,dataType]) 
    else:
        minValue = min(data[end-1,:,dataType])
        maxValue = max(data[beg,:,dataType]) 
    interval = maxValue-minValue
    plt.ylim(minValue-interval*0.2, maxValue+interval*0.2)
    plt.show()
    return 

#start = 2000       
#end=epochs    
#plot_for_given_alpha(accuracyHistory,start,end, 0, "Accuracy", True)
#plot_for_given_alpha(accuracyHistory,start,end, 1, "Accuracy", True)
#plot_for_given_alpha(accuracyHistory,start,end, 2, "Accuracy", True)
#
#plot_for_given_alpha(lossHistory,start,end, 0, "Loss", False)
#plot_for_given_alpha(lossHistory,start,end, 1, "Loss", False)
#plot_for_given_alpha(lossHistory,start,end, 2, "Loss", False)

#plot_alpha_comparison(accuracyHistory, start, end, 0, "Accuracy", True)
#plot_alpha_comparison(accuracyHistory, start, end, 1, "Accuracy", True)
#plot_alpha_comparison(accuracyHistory, start, end, 2, "Accuracy", True)

#plot_alpha_comparison(lossHistory, start, end, 0, "Loss", False)
#plot_alpha_comparison(lossHistory, start, end, 1, "Loss", False)
#plot_alpha_comparison(lossHistory, start, end, 2, "Loss", False)

################################################################
#
#plot_for_given_reg(accuracyHistory_reg,start,end, 0, "Accuracy", True)
#plot_for_given_reg(accuracyHistory_reg,start,end, 1, "Accuracy", True)
#plot_for_given_reg(accuracyHistory_reg,start,end, 2, "Accuracy", True)

#plot_reg_comparison(accuracyHistory, start, end, 0, "Accuracy", True)
#plot_reg_comparison(accuracyHistory, start, end, 1, "Accuracy", True)
#plot_reg_comparison(accuracyHistory, start, end, 2, "Accuracy", True)
#
#plot_reg_comparison(lossHistory, start, end, 0, "Loss", False)
#plot_reg_comparison(lossHistory, start, end, 1, "Loss", False)
#plot_reg_comparison(lossHistory, start, end, 2, "Loss", False)

##0 is 0.001, 1 is 0.1, 2 is 0.5
#reg_number=0
#print(lossHistory_reg[epochs-1,reg_number,0]) #Train
#print(lossHistory_reg[epochs-1,reg_number,1]) #Valid
#print(lossHistory_reg[epochs-1,reg_number,2]) #Test

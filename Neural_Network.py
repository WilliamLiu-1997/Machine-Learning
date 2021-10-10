import math
import random
import copy
import time

class NN:

    #initialize the model
    #hidden should be passed a list 
    #e.g. hidden=[15,8] means model will have two hidden layers, first hidden layer has 15 nodes and second one has 8 nodes
    #attributes means the number of attributes
    #classes means the number of classes in the cllasification tasks
    #activation means the activation function, mush be "ReLU" or "Sigmoid"
    def __init__(self, hidden, attributes, classes, activation):

        #Size store the number of nodes in each layer
        self.size = [attributes] + hidden + [classes]
        #value in each node in each layer
        self.nodes = []
        #value of each weight
        self.weights = []
        #value of each bias
        self.bias = []
        #momentum is used to help get rid of local minima
        #value of momentum of each weight
        self.momentums_w = []
        #value of momentum of each bias
        self.momentums_b = []
        #number of classes to classified
        self.classes = classes
        #activation function for hidden layers ("ReLU" or "Sigmoid")
        self.activation=activation

        #expand the size of nodes in each layer
        for layer in range(len(self.size)):
            self.nodes.append([0.0] * self.size[layer])
        #expand the size of bias in each layer
        for layer in range(1, len(self.size)):
            self.bias.append([0.0] * self.size[layer])
            self.momentums_b.append([0.0] * self.size[layer])
        #initialize biases with a random value between -1 and 1
        for layer in range(1, len(self.size)):
            for node in range(self.size[layer]):
                self.bias[layer-1][node] = random.uniform(-1, 1)
        #expand the size of weights in each layer
        for layer in range(len(self.size) - 1):
            matrix = []
            for _ in range(self.size[layer]):
              matrix.append([0.0] * self.size[layer + 1])
            self.weights.append(copy.deepcopy(matrix))
            self.momentums_w.append(copy.deepcopy(matrix))
        #initialize weights with a random value between -1 and 1
        for layer in range(len(self.size) - 1):
            for m in range(self.size[layer]):
                for n in range(self.size[layer + 1]):
                    self.weights[layer][m][n] = random.uniform(-1, 1)

    #define the activation function(Sigmoid) for hidden layers
    def Sigmoid(self, input):
        if input >= 0:
            return 1.0 / (1.0 + math.exp(-input))
        else:
            return math.exp(input) / (1.0 + math.exp(input))

    #define the activation function(ReLU) for hidden layers
    def ReLU(self, input):
      if input>0:
        return input
      else:
        return 0

    def Activation(self,input):
      if self.activation=="ReLU":
        return self.ReLU(input)
      if self.activation=="Sigmoid":
        return self.Sigmoid(input)

    #define the Derivative of activation function(Sigmoid) for hidden layers
    def Derivative_Sigmoid(self, input):
        return input * (1 - input)

    #define the Derivative of activation function(ReLU) for hidden layers 
    def Derivative_ReLU(self, input):
        if input>0:
          return 1
        else: 
          return 0

    def Derivative_Activation(self,input):
      if self.activation=="ReLU":
        return self.Derivative_ReLU(input)
      if self.activation=="Sigmoid":
        return self.Derivative_Sigmoid(input)
        
    #define the activation function(Softmax) for output layers
    def Softmax(self, inputs):
        sum = 0
        for i in inputs:
            sum += math.exp(i)
        return [math.exp(input)/sum for input in inputs]

    #define the loss function(cross entropy) for output
    def Loss(self, outputs, labels):
        results = []
        for i in range(len(labels)):
            results.append(-labels[i]*math.log(1e-6+outputs[i])/len(labels))
        return results

    #define the derivative for output layers (d(CrossEntropy(target,SoftMax(x))/dx)
    def Derivative_Loss(self, output, label):
        return output-label


    #forward propagate
    def Forward(self, inputs):
        for i in range(self.size[0]):
            self.nodes[0][i] = inputs[i]
        last_layer = len(self.size)-1
        for layer in range(1, last_layer):
            for node in range(self.size[layer]):
                total = 0.0
                for i in range(self.size[layer - 1]):
                    total += self.nodes[layer - 1][i] * self.weights[layer - 1][i][node]
                total += self.bias[layer-1][node]
                self.nodes[layer][node] = self.Activation(total)

        for node in range(self.size[last_layer]):
            total = 0.0
            for i in range(self.size[last_layer - 1]):
                total += self.nodes[last_layer - 1][i] * self.weights[last_layer - 1][i][node]
            total += self.bias[last_layer-1][node]
            self.nodes[last_layer][node] = total
        self.nodes[last_layer] = self.Softmax(self.nodes[last_layer])
        #return the result
        return self.nodes[-1]

    #back propagate
    def Back(self, input, label, learn_r, mom):

        self.Forward(input)
        deltas = []

        #error back propagate
        for layer in reversed(range(1, len(self.size))):
            if layer == len(self.size) - 1:
                deltas.insert(0,[0.0] * self.size[layer])
                for i in range(self.size[layer]):
                    # Derivative of error for each outputs
                    error = self.Derivative_Loss(self.nodes[layer][i], label[i])
                    deltas[0][i] = error
            else:
                deltas.insert(0,[0.0] * self.size[layer])
                # transmit the Derivative of error to each node
                for i in range(self.size[layer]):
                    error = 0.0
                    for j in range(self.size[layer + 1]):
                        error += deltas[1][j] * self.weights[layer][i][j]
                    deltas[0][i] = self.Derivative_Activation(self.nodes[layer][i]) * error

        #Update weights
        for layer in reversed(range(1, len(self.size))):
            for i in range(self.size[layer - 1]):
                for j in range(self.size[layer]):
                    delta_w = - deltas[layer - 1][j] * self.nodes[layer - 1][i] * learn_r + mom * self.momentums_w[layer - 1][i][j]
                    self.weights[layer - 1][i][j] += delta_w
                    self.momentums_w[layer - 1][i][j] = delta_w
        #Update biases
        for layer in reversed(range(1, len(self.size))):
            for i in range(self.size[layer]):
                delta_b = - deltas[layer - 1][i] * learn_r + mom * self.momentums_b[layer - 1][i]
                self.bias[layer - 1][i] += delta_b
                self.momentums_b[layer - 1][i] = delta_b

        #calculate new error
        self.Forward(input)
        error = 0.0
        result = self.Loss(self.nodes[len(self.nodes) - 1], label)
        for i in range(len(label)):
            error += result[i]
        return error

    #Define the training process
    def Train(self, inputs, labels, max_round, learn_r, mom, min_error):
        error = 0
        stay = 0
        last_error=10e10
        print("Learn Rate--->{0}".format(learn_r))
        for i in range(max_round):
            error = 0
            sample = random.sample(range(0, len(inputs)), len(inputs))
            #adaptive learning rate
            if stay >= 10:
                learn_r = learn_r * 0.9
                stay = 0
                print()
                print("Update Learn Rate--->{0}".format(learn_r))
            for j in sample:
                label = labels[j]
                input = inputs[j]
                error += self.Back(input, label, learn_r, mom)
            error /= len(sample)
            print("Training No.{: <5d}  Error:{: .10f}".format(i+1, error))

            if error > last_error * 1.01:
                stay += 10
            elif error > last_error * 0.99:
                stay += 1
            else:
                stay = 0

            last_error = error
            #early stop when error<min_error
            if error < min_error:
                print("Training End!      Error:{: .10f}".format(error))
                return
        print("Training End!      Error:{: .10f}".format(error))
        return

    #one-hot encode the label
    def one_hot(self, input):
        one_hot_labels = []
        for i in range(len(input)):
          label = []
          for j in range(self.classes):
              label.append(0)
          label[input[i]] = 1
          one_hot_labels.append(label)
        return one_hot_labels

    #the whole training process, one-hot encoding labels, training
    def fit(self, cases, labels, lr=0.1, max_iter=1000, mom=0.1, min_error=1e-3):
        start = time.time()
        print("Start training "+str(len(labels))+" samples...")
        one_hot_labels = self.one_hot(labels)
        self.Train(cases, one_hot_labels, max_iter, lr, mom, min_error)
        end = time.time()
        print("Elapsed time: "+ str(round(end - start,2))+"s")
        print()
        # return [self.size, self.nodes, self.weights, self.bias]


    #predict multiple inputs
    def predict(self, inputs):
        outputs = []
        for case in range(len(inputs)):
            result = self.Forward(inputs[case])
            outputs.append(result.index(max(result)))
        return outputs


    #verify the training results
    def test(self, cases, labels):
        right = 0
        print("Testing "+str(len(labels))+" samples...")
        print("Sample    Correct      Prediction     Label")
        results=self.predict(cases)
        for case in range(len(cases)):
            result = results[case]
            label = labels[case]
            if result == label:
                right += 1
                print("No.{:<5d}    *           {:=4d}         {:=4d}".format(case+1, result, label))
            else:
                print("No.{:<5d}                {:=4d}         {:=4d}".format(case+1, result, label))
        print("Accuracy : {: .2f}%".format(100 * right / len(cases)))
        print()











from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
Data = load_wine()

#load_breast_cancer
#load_iris
#load_wine
#load_digits


X, Y = Data['data'], Data['target']
x = X.tolist()
y = Y.tolist()

#get 80% of the dataset to be the training dataset and reset 20% to be the test dataset
train_sample_x,test_sample_x,train_sample_y,test_sample_y=train_test_split(x,y,test_size=0.2)

#transform the data
#for each attribute x= (x - u) / s  ---->u=mean and s=stanard deviation
scaler = StandardScaler()
scaler.fit(train_sample_x)
train_sample_x = scaler.transform(train_sample_x)
test_sample_x = scaler.transform(test_sample_x)


#define the network
#hidden:have two hidden layers, one with 15 nodes and one with 8 nodes
#attributes:number of attribute(used to set input nodes number)
#classes:number of classes(used to set output nodes number)
network = NN(hidden = [15,8], attributes = X.data.shape[1], classes = max(Y)+1, activation="Sigmoid")
#Train the model using train dataset
network.fit(train_sample_x, train_sample_y, lr=0.1, max_iter=100, mom=0.1, min_error=1e-3)
#Test thge model using test dataset
network.test(test_sample_x, test_sample_y)
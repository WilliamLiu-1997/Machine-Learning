import numpy as np
import random
import math

class Decision_Tree:
    def __init__(self):
        self.tree={}
        self.major_label=False

    def cal_info_entropy(self, data):
        if len(data)==0:
          return 0
        partition = 1/len(data)
        label_prob={}
        for record in data:
            label_prob[record[-1]]=label_prob.get(record[-1],0)+partition

        info_entropy=sum([-label_prob[label]*math.log(label_prob[label]+1e-6,2) for label in label_prob])
        return info_entropy

    def get_branches(self,data,feature,mean):
        branch_greater=[]
        branch_smaller=[]
        for input in data:
            if input[feature]>mean:
                branch_greater.append(input)
            if input[feature] <= mean:
                branch_smaller.append(input)
        return branch_greater,branch_smaller

    def choose_feature(self, data):
        feature_num=len(data[0])-1
        base_info_entropy = self.cal_info_entropy(data)
        best_info_gain = 0
        best_feature=-1
        
        features=[record[:-1] for record in data]

        means=np.mean(features,axis=0)
        for feature in range(feature_num):
            entropy=0
            mean=round(means[feature],2)
            branches_greater,branches_smaller=self.get_branches(data, feature, mean)
            entropy += len(branches_greater)/len(data)*self.cal_info_entropy(branches_greater)
            entropy += len(branches_smaller)/len(data)*self.cal_info_entropy(branches_smaller)
            info_gain = base_info_entropy-entropy
            if info_gain > best_info_gain:
                best_info_gain=info_gain
                best_feature=feature
                best_feature_mean=mean
        if best_feature==-1:
          return False,False
        else:
          return best_feature,best_feature_mean

    def get_major_label(self, labels):
        label_count={}
        for label in labels:
            label_count[label]=label_count.get(label,0)+1
        return max(label_count, key=label_count.get)
    
    def fit(self, data, features_name):
        inputs = [value[:-1] for value in data]
        labels = [value[-1] for value in data]

        if self.major_label is False:
          self.major_label=self.get_major_label(labels)

        if len(inputs)==0:
            return self.major_label
        if len(set(labels))==1:
            return labels[0]
        if len(inputs[0])==0:
            return self.get_major_label(labels)
        

        best_feature,value=self.choose_feature(data)
        if best_feature is False:
          return self.get_major_label(labels)
        best_feature_name = features_name[best_feature]
        tree = {best_feature_name: {}}
        branches_greater,branches_smaller=self.get_branches(data, best_feature, value)
        tree[best_feature_name]["> "+str(value)] = self.fit(branches_greater, features_name)
        tree[best_feature_name]["<= "+str(value)] = self.fit(branches_smaller, features_name)
        
        self.tree=tree
        return tree
    
    def predict(self,tree,input,features_name):
        if not isinstance(tree,dict):
          return tree
        feature_name=list(tree.keys())[0]
        feature = features_name.index(feature_name)
        child_tree = tree[feature_name]
        for key in child_tree.keys():
            compare=key.split(" ")[0]
            value=float(key.split(" ")[-1])
            if compare==">" and input[feature]>value or compare == "<=" and input[feature] <= value:
                result = self.predict(child_tree[key], input, features_name)
        return result
    
    def test(self,inputs, features_name):
        right = 0
        print("Testing "+str(len(inputs))+" samples...")
        print("Sample    Correct        Prediction           Label")
        for case in range(len(inputs)):
            input = inputs[case][:-1]
            label = inputs[case][-1]
            predict_value = self.predict(self.tree, input, features_name)
            if predict_value == label:
                right += 1
                print(
                    "No.{:<5d}     *           {: ^10}         {: ^10}".format(case+1, predict_value, label))
            else:
                print("No.{:<5d}                 {: ^10}         {: ^10}".format(case+1, predict_value, label))
        print("Accuracy : {: .2f}%".format(100 * right / len(inputs)))
        print()







from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
Data = load_iris()
#load_breast_cancer
#load_iris
#load_wine
#load_digits
X, Y = Data['data'], Data['target']
target_name=Data['target_names'].tolist()
x = X.tolist()
y = Y.tolist()

if "feature_names" in Data.keys():
  feature_names = Data.feature_names
else:
  feature_names=[]
  for i in range(len(x[0])):
    feature_names.append(i)

if type(feature_names) is np.ndarray:
  feature_names=feature_names.tolist()


for i in range(len(x)):
  x[i].append(target_name[y[i]])

train_sample,test_sample=train_test_split(x,test_size=0.2)

Decision_Tree = Decision_Tree()
Decision_Tree.fit(train_sample, feature_names)
Decision_Tree.test(test_sample, feature_names)







from anytree import RenderTree, Node
from anytree.exporter import DotExporter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
nodeList=[Node("root")]
def transfer_tree(tree,parent,No):
  No+=1
  if not isinstance(tree,dict):
    nodeList.append(Node(str(No)+" | "+str(tree),parent=parent))
    return No
  nodeName=list(tree.keys())[0]
  for name in tree[list(tree.keys())[0]]:
    nodeList.append(Node(str(No)+" | "+str(nodeName)+name,parent=parent))
    No=transfer_tree(tree[list(tree.keys())[0]][name],nodeList[-1],No)
  return No

transfer_tree(Decision_Tree.tree,nodeList[0],0)
img=DotExporter(nodeList[0],nodeattrfunc=lambda node: "shape=box").to_picture("tree.png")
img = mpimg.imread('tree.png')
plt.figure(figsize = (50,50)) 
imgplot = plt.imshow(img)
plt.show()
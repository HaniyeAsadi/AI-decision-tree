import pandas as pd
from decision_tree import DecisionTree

data = pd.read_csv("./diabetes.csv")
train_per = 0.7
train_num = int(len(data) * train_per)
test_num = len(data) - train_num
training_data = pd.read_csv("./diabetes.csv", skipfooter=test_num)
testing_data = pd.read_csv("./diabetes.csv", skiprows = lambda x: train_num<x<=test_num, nrows=len(data)-train_num)

dec_tree = DecisionTree(max_depth=7)
dec_tree.fit(training_data= training_data)

sequence = dec_tree._branch_criteria(train_data=training_data)
branch_cr = []
for sq in sequence:
    branch_cr.append(sq[0])
branch_cr.reverse()
print("The order of attributes to branching tree is : ")
print(' , '.join(branch_cr))
correct_predict, wrong_predict = 0, 0
def _test_data(tree, data):
    global correct_predict, wrong_predict
    out = data["Outcome"]
    node = tree.root
    for i in branch_cr:
        threshold = node.threshold
        if(node.value != None):
            if(float(node.value) == out):
                correct_predict +=1
            else:
                wrong_predict +=1
            break
        elif(data[i] >= threshold and node.left != None):
            node = node.left
        elif(data[i] < threshold and node.right != None):
            node = node.right

for data in range (len(testing_data)):
    t = testing_data.iloc[data]
    _test_data(tree=dec_tree, data=t)

correct_percentage = (correct_predict / test_num) * 100
correct_percentage = float('%.3f'%correct_percentage)
print("This program diagnoses diabetes with a probability of "+ str(correct_percentage) + "%")
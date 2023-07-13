import numpy as np
import math
from collections import Counter
from operator import itemgetter

root_entropy = 0
info_gain = []
pos_value, neg_value = 1, 0
root_Node =0
num=len(info_gain)-1
class Node :
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature#ملاک دسته بندی
        self.threshold = threshold# مقداری که برای دسته بندی کردن ملاک قرار داده ایم
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree :
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def fit(self, training_data):
        # بدست آوردن انتروپی بر اساس همه ستون های دیتا
        self.entropy(training_data, 0)
        self._information_gain(training_data)
        # شروع درخت: ساخت ریشه
        self.root = self._make_root(training_data)
        return self.root

    def _make_root(self, train_data):
        return self._make_tree(train_data=train_data, depth=0)

    def _make_tree(self, train_data, depth):
        global num, info_gain
        count = Counter(train_data["Outcome"])
        pos = count.most_common(1)[0][1]
        en1 = (pos) / len(train_data["Outcome"])
        en2 = 1 - float(en1) 
        if(depth >= self.max_depth or en1 < 0.005 or en2 < 0.005):
            leaf_node = self._most_common_data(train_data["Outcome"])
            return Node(value= leaf_node)
        if(depth < self.max_depth):
            criteria = info_gain[num]
            num-=1
            left_data = train_data.loc[train_data[criteria[0]] >= criteria[1]]
            right_data = train_data.loc[train_data[criteria[0]] < criteria[1]]
            if(len(left_data)==0 and len(right_data)!=0):
                left_child = None
                right_child = self._make_tree(right_data, depth=depth+1)
                num+=1
            elif(len(right_data)==0 and len(left_data)!=0):
                right_child = None
                left_child = self._make_tree(left_data, depth=depth+1)
                num+=1
            else:
                left_child = self._make_tree(left_data, depth=depth+1)
                right_child = self._make_tree(right_data, depth=depth+1)
                num+=1
            return Node(feature=criteria[0], threshold=criteria[1], left=left_child, right=right_child, value=None)

    def _most_common_data(self, outcome_data):
        count = Counter(outcome_data)
        return count.most_common(1)[0][0]

    def entropy(self, data, depth):
        total = len(data)
        pos_num = np.count_nonzero(data[data.columns[-1]]==1, axis=0)
        neg_num = total - pos_num
        p_pos, p_neg = pos_num/total, neg_num/total
        en=0
        if depth==0:
            global root_entropy 
            root_entropy = -1 * (p_pos * math.log(p_pos, 2) + p_neg * math.log(p_neg, 2))
            root_entropy = float('%.5f'%root_entropy)
        else:
            en = -1 * (p_pos * math.log(p_pos, 2) + p_neg * math.log(p_neg, 2))
            en = float('%.5f'%en)
            return en 

    def _information_gain(self, train_data, depth=1):
        #اسم ستون های دیتابیس
        global info_gain, root_entropy
        columns = train_data.columns[:-1]
        #بدست آوردن انتروپی هر ستون
        for cl in columns:
            avg = (np.min(train_data[cl]) + np.max(train_data[cl])) /2
            avg = float('%.3f'%avg)
            greater_data = train_data.loc[train_data[cl] >= avg]
            greater_entropy = self.entropy(greater_data, depth=depth)
            greater_entropy = float('%.5f'%greater_entropy)
            lower_data = train_data.loc[train_data[cl] < avg]
            lower_entropy = self.entropy(lower_data, depth=depth)
            lower_entropy = float('%.5f'%lower_entropy)
            p_greater, p_lower = len(greater_data)/len(train_data[cl]), len(lower_data)/len(train_data[cl])
            p_greater, p_lower = float('%.2f'%p_greater), float('%.2f'%p_lower)
            cl_entropy = p_greater*greater_entropy + p_lower*lower_entropy
            cl_entropy = float('%.5f'%cl_entropy)
            ig = root_entropy - cl_entropy
            ig = float('%.5f'%ig)
            tpl = (cl, avg, ig)
            info_gain.append(tpl) 
        info_gain = sorted(info_gain, key=itemgetter(2))     

    def _branch_criteria(self, train_data):
        global info_gain
        return info_gain
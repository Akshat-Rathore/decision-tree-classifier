import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt



INF = 1000
TARGET_COL = 13     # Target column no
ROWS = 270          # No of data rows
COLS = 14           # No of columns
SPLIT_RATIO = 0.8   # Ratio for spliting traning/test data
MAX_DEPTH = 15      # Maximum depth for growing the tree
RANDOM_SPLITS = 10  # No of random splits on data set considered

# X-sq threshold values for p = 0.05 significance level
# (Key, value) <- (degree of freedom, threshold)
X_SQ = {
    1: 3.84,
    2: 5.99,
    3: 7.81,
    4: 9.49,
    5: 11.07
}
# p = 0.01
# X_SQ = {
#     1: 6.63,
#     2: 9.21,
#     3: 11.34,
#     4: 13.28,
#     5: 15.09
# }

# p = 0.10
# X_SQ = {
#     1: 2.71,
#     2: 4.61,
#     3: 6.25,
#     4: 7.78,
#     5: 9.24
# }


def get_random():
    import random
    randomlist = random.sample(range(1, ROWS), int(ROWS * SPLIT_RATIO))
    return randomlist

class Attribute:
    def __init__(self, pid, name, dtype, ntype, values=[]):
        self.name = name
        self.id = pid
        self.type = dtype     # discrete or continuous(d or c)
        self.ntype = ntype   # number of discrete values (0 for c)
        self.values = values  # list of values stored  

    def __str__(self):
        return self.name

class Node:
    def __init__(self, data, depth, pnode, id, method = "GINI"):
        self.id = id               # Node-ID
        self.depth = depth         # depth of node
        self.attribute = None      # attribute based on which node data is furthur classified (None for pure node)
        self.pnode = pnode         # parent node
        self.childnodes = {}       # Dictionary of childnodes (key -> value of attribute and value -> child node)
        self.data = data           # part of data this node represents
        self.bestsplitc = None     # for continuous values, None for discrete values
        self.gini = None           # Gini impurity of the node
        self.entropy = None        # Entropy of the node
        self.ginigain = None       # Gini gain
        self.infogain = None       # information gain
        self.label = None          # Label <-- For pure Node it's 1 or, 2; Otherwise it's None

        if method == "GINI" and len(data) != 0:
            self.gini = Node.gini_calculator(data)
            # If pure node initialize the label
            if self.gini == 0:
                self.label = data[0][TARGET_COL]
        if method == "ENTROPY" and len(data) != 0:
            self.entropy = Node.entropy_calculator(data)
            # If pure node initialize the label
            if self.entropy == 0:
                self.label = data[0][TARGET_COL]

    def get_most_freq_label(self):
        one = 0
        for d in self.data:
            one += (d[TARGET_COL] == 1)
        if one >= len(self.data) - one:
            return 1.0
        return 2.0

    def count_pos_samples(self):
        res = 0
        for d in self.data:
            res += (d[TARGET_COL] == 1.0)
        return res

    @staticmethod
    def gini_calculator(data):
        p = len([d for d in data if d[TARGET_COL] == 1]) / len(data)
        return  2 * p * (1 - p) 

    @staticmethod
    def entropy_calculator(data):
        p = len([d for d in data if d[TARGET_COL] == 1]) / len(data)
        if p == 0.0 or p == 1.0:
            return 0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    @staticmethod
    def bestSplitGini(node, data, attrbs):    # find best split according to gini
        max = -INF           # max gain so far
        bestsplit = None     #  bestsplitting attribute so far.. initialized to None
        bestsplitc = None
        current_gini = node.gini
        for attrb in attrbs:
            if attrb.type == 'c':
                valueset = []
                pos = attrb.id
                for i in data:
                    valueset.append(i[pos])
                valueset.sort()
                l = len(valueset)
                split_pt = None
                maxc = -INF
                for i in range(l):
                    if i + 1 < l and valueset[i] != valueset[i + 1]:
                        j = (valueset[i] + valueset[i + 1]) / 2     # Split point
                        data_left = [d for d in data if d[attrb.id] <= j]            
                        data_right = [d for d in data if d[attrb.id] > j]
                        gini_left = Node.gini_calculator(data_left)
                        gini_right = Node.gini_calculator(data_right)
                        gini = (len(data_left) / len(data)) * gini_left + (len(data_right) / len(data)) * gini_right
                        gain = current_gini - gini
                        if maxc < gain:
                            maxc = gain
                            split_pt = j
                if split_pt == None:
                    # Attribute has uniform value over all data 
                    # So, gini-impurity for this attr = node's gini-impurity
                    if maxc < 0:
                        maxc = 0
                        split_pt = valueset[0] 
                if max < maxc:
                    max = maxc
                    bestsplit = attrb
                    bestsplitc = split_pt
            else:       
                gini = 0.0
                for v in attrb.values:
                    # gets dataset with value v for attribute attrb
                    datav = [d for d in data if d[attrb.id] == v]            
                    if len(datav):
                        gini += Node.gini_calculator(datav) * (len(datav) / len(data))
                ginigain = current_gini - gini
                if max < ginigain:
                    max = ginigain
                    bestsplit = attrb

        return_tupple = None
        # Return type --> <Best splitting attr, max gain, split point (in case of cont. or, -1 for discrete)
        if bestsplit.type == 'c':
            return_tupple = (bestsplit, max, bestsplitc)
        else:
            return_tupple = (bestsplit, max, -1)
        return return_tupple

    @staticmethod
    def bestSplitEntropy(node, data, attrbs):            # find best split according to entropy
        max = -INF
        bestsplit = None
        bestsplitc = None
        current_entropy = node.entropy
        for attrb in attrbs:
            if attrb.type == 'c':
                valueset = []
                pos = attrb.id
                for i in data:
                    valueset.append(i[pos])
                valueset.sort()
                l = len(valueset)
                maxc = -INF
                split_pt = None
                for i in range(l):
                    if i + 1 < l and valueset[i] != valueset[i + 1]:
                        j = (valueset[i] + valueset[i + 1]) / 2     # Split point
                        entropysum = 0.0
                        data_left = [d for d in data if d[attrb.id] <= j]            
                        data_right = [d for d in data if d[attrb.id] > j]
                        entropy_left = Node.entropy_calculator(data_left)
                        entropy_right = Node.entropy_calculator(data_right)
                        entropysum = (len(data_left) / len(data)) * entropy_left + (len(data_right) / len(data)) * entropy_right
                        infogain = current_entropy - entropysum
                        if maxc < infogain:
                            maxc = infogain
                            split_pt = j

                if split_pt == None:
                    # Attribute has uniform value over all data 
                    # So, entropy for this attr = node's entropy
                    if maxc < 0:
                        maxc = 0
                        split_pt = valueset[0] 
                if max < maxc:
                    max = maxc
                    bestsplit = attrb
                    bestsplitc = split_pt
            else:
                entropysum = 0.0
                for v in attrb.values:
                    # gets dataset with value v for attribute attrb
                    datav = [d for d in data if d[attrb.id] == v]            
                    if len(datav):
                        entropysum += Node.entropy_calculator(datav) * (len(datav) / len(data))
                gain = current_entropy - entropysum
                if max < gain:
                    max = gain
                    bestsplit = attrb
        
        return_tupple = None
        # Return type --> <Best splitting attr, max gain, split point (in case of cont. or, -1 for discrete)
        if bestsplit.type == 'c':
            return_tupple = (bestsplit, max, bestsplitc)
        else:
            return_tupple = (bestsplit, max, -1)
        return return_tupple

    def __str__(self):
        if self.gini != None:
            if self.pnode == None:
                return f"<Root, Node-ID-{self.id}, data-{len(self.data)}, attr-{self.attribute}, gini-{self.gini}, label-{self.label}>"
            return f"<parent-{self.pnode.id}, Node-ID-{self.id}, data-{len(self.data)}, attr-{self.attribute}, gini-{self.gini}, label-{self.label}>"
        else:
            if self.pnode == None:
                return f"<Root, Node-ID-{self.id}, data-{len(self.data)}, attr-{self.attribute}, entropy-{self.entropy}, label-{self.label}>"
            return f"<parent-{self.pnode.id}, Node-ID-{self.id}, data-{len(self.data)}, attr-{self.attribute}, entropy-{self.entropy}, label-{self.label}>" 
    

class DecisionTree:
    def __init__(self, maxdepth = None):
        self.maxdepth = maxdepth
        self.count = 0
        self.depth = 0
        self.root = None
        self.start_id = 0

    def generateDT(self, attrs, data, method = "GINI", depth = 0, node = None):
        self.depth = max(self.depth, depth)
        # Initialize the root if not None
        if self.root == None:
            self.root = Node(data, 0, None, self.start_id, method)
            self.start_id += 1
            self.count += 1
            node = self.root
        
        # Base cases
        # Base case - 1 -> Pure Node
        if method == "GINI" and node.gini == 0:
            return
        if method == "ENTROPY" and node.entropy == 0:
            return

        # Base Case - 2 -> Attribute list is empty
        if len(attrs) == 0:
            # Initialize label with most frequent label in data
            node.label = node.get_most_freq_label()
            return

        # Base Case - 3 -> Depth limit (if any)
        if self.maxdepth is not None and node.depth == self.maxdepth:
            # Initialize label with most frequent labelling in data
            node.label = node.get_most_freq_label()
            return

        # Get the best attribute that classifies the current node's data
        bestattr = None 
        gain = None 
        split_point = None 
        if method == "GINI":
            (bestattr, gain, split_point) = Node.bestSplitGini(node, data, attrs)
            node.ginigain = gain
        else:
            (bestattr, gain, split_point) = Node.bestSplitEntropy(node, data, attrs)
            node.infogain = gain
        
        node.attribute = bestattr   
        # Create branches and continue recursion
        attrs.remove(bestattr)  # Remove the bestatrr since it's not usable in current node's subtree
        if bestattr.type == 'c':
            # Only two children
            self.count += 2
            left_partition = [d for d in data if d[bestattr.id] <= split_point]
            right_partition = [d for d in data if d[bestattr.id] > split_point]
            left_child = Node(left_partition, depth + 1, node, self.start_id, method)
            self.start_id += 1
            right_child = Node(right_partition, depth + 1, node, self.start_id, method)
            self.start_id += 1
            node.bestsplitc = split_point
            node.childnodes[0] = left_child                                   
            node.childnodes[1] = right_child
            if len(left_partition) == 0:
                left_child.label = node.get_most_freq_label()
            else:
                self.generateDT(attrs, left_partition, method, depth + 1, left_child)

            if len(right_partition) == 0:
                right_child.label = node.get_most_freq_label()
            else:
                self.generateDT(attrs, right_partition, method, depth + 1, right_child)
        else:
            for v in bestattr.values:
                self.count += 1                                                                 # Increase total node count
                partition = [d for d in data if d[bestattr.id] == v]                            # partition of data having bestattr value = v in data
                child = Node(partition, depth + 1, node, self.start_id, method)                 # Create new child node for this partition
                self.start_id += 1
                node.childnodes[v] = child                                                      # Store the branch information
                if len(partition) == 0: 
                    # If partition is empty initialize the label of child as the most freq label of parent node
                    child.label = node.get_most_freq_label()
                else:
                    self.generateDT(attrs, partition, method, depth + 1, child)     # Continue recursion for child node with data = partition (non-empty)


        attrs.append(bestattr)    # Reinsert bestattr

    def post_pruning(self):
        leaves = []      # Stores all the leaf nodes
        q = deque()
        q.append(self.root)
        while len(q):
            node = q.popleft()
            for (key, child) in node.childnodes.items():
                if child.label is not None:
                    leaves.append(child)
                else:
                    q.append(child)

        # Keep on doing pruning until no splitting is possible
        while True:
            # Select a leaf node and it's parent that can be split
            par = None
            for leaf in leaves:
                pnode = leaf.pnode
                if pnode == None:
                    continue
                split = True
                for child in pnode.childnodes.values():
                    if child.label == None:
                        split = False
                        break

                if not split:
                    continue

                # Count of pos and neg samples in pnode's data
                pos_sample = pnode.count_pos_samples()
                neg_sample = len(pnode.data) - pos_sample

                # Statistic for X-sq test
                K = 0
                # Calculate the value of K 
                for child in pnode.childnodes.values():
                    # True Count of pos and neg sample for child node
                    true_pos_sample = child.count_pos_samples()
                    true_neg_sample = len(child.data) - true_pos_sample

                    # Expected count of pos and neg sample for child node
                    exp_pos_sample = pos_sample * len(child.data) / len(pnode.data)
                    exp_neg_sample = neg_sample * len(child.data) / len(pnode.data)

                    if exp_pos_sample:
                        K += (true_pos_sample - exp_pos_sample) * (true_pos_sample - exp_pos_sample) / exp_pos_sample
                    if exp_neg_sample:
                        K += (true_neg_sample - exp_neg_sample) * (true_neg_sample - exp_neg_sample) / exp_neg_sample

                # Degree of X-sq distribution
                deg_X_sq = len(pnode.childnodes) - 1
 
                if K < X_SQ[deg_X_sq]:
                    par = pnode
                    break
            # If par is none no furthur splitting is possible
            if par is None:
                break
            else:
                # Reduce the parent to a leaf node
                # Remove every children of par from leaf  
                for child in par.childnodes.values():
                    leaves.remove(child)

                par.childnodes = {}
                par.bestsplitc = None
                par.ginigain = None      
                par.infogain = None    
                par.label = par.get_most_freq_label()
                par.attribute = None   
                leaves.append(par)

        q = deque()
        q.append(self.root)
        self.count = 1
        self.depth = 0
        while len(q):
            node = q.popleft()
            for (key, child) in node.childnodes.items():
                q.append(child) 
                self.count += 1
                self.depth = max(self.depth, child.depth)          
    # Evaluates a single data
    def evaluate(self, data):
        node = self.root
        # Keep traversing the DT until reaching a leaf or, pure node
        while node.label == None:
            attr_val = data[node.attribute.id]  # Corresponding attr value for data
            if node.attribute.type == 'c':
                if attr_val <= node.bestsplitc:
                    node = node.childnodes[0]   # If less than go to left child
                else:                           # Else go to right child
                    node = node.childnodes[1]
            else:
                node = node.childnodes[attr_val]    # Go to the branch having the attr_val
        return node.label == data[TARGET_COL]

    # Measures the accuracy rate for a given test data
    # Accuracy rate <-- No of correct classifications / Total data size
    def accuracy_test(self, data):
        correct = 0
        for d in data:
            correct += self.evaluate(d)
        return correct / len(data)

    # Prints the DT in level order 
    def print_tree(self):
        q = deque()
        q.append((None, self.root))
        while len(q):
            l = len(q)
            # Pop each node in the current level and insert children of them
            for _ in range(l):
                (branch, node) = q.popleft()
                print(f"<branch-{branch}, ", end = '')
                print(node, end = '')
                print(">")
                for (key, child) in node.childnodes.items():
                    if node.attribute.type == 'c':
                        if key == 0:
                            q.append((f"<= {node.bestsplitc}", child))
                        else:
                            q.append((f"> {node.bestsplitc}", child))
                    else:
                        q.append((key, child))
            print("=========================================================================================")

    def createIMG(self,name):
        if self.root.gini!=None:
            f=open("./dot/"+name+"Gini.dot","w")
        else:
            f=open(name+"entropy.dot","w")
        f.write("digraph G {\nrankdir=LR \nnode [shape=rectangle]\n")
        f.write("1.0[taillabel=\"Target Value\"];\n2.0[taillabel=\"Target Value\"]; ")
        q = deque()
        q.append((None, self.root))
        while len(q):
            l = len(q)
            for _ in range(l):
                (branch, node) = q.popleft()
                if node.gini!=None:
                    if node.id!=0 and node.label==None:
                        f.write(f"\n{node.id}[label=< {node.attribute} <BR/>\n<FONT POINT-SIZE=\"10\"> n(Data): {len(node.data)}\n Gini: {node.gini} </FONT> >];\n")
                    elif node.label==None:
                        f.write(f"\n{node.id}[xlabel=\"ROOT\" , label=< {node.attribute} <BR/>\n<FONT POINT-SIZE=\"10\"> n(Data): {len(node.data)}\n Gini: {node.gini} </FONT> >];\n")
                    else:
                        pass
                else:
                    if node.label==None:
                        f.write(f"\n{node.id}[label=< {node.attribute}<BR/>\n<FONT POINT-SIZE=\"10\"> n(Data): {len(node.data)}\n Entropy: {node.entropy}</FONT>>];\n")
                for (key, child) in node.childnodes.items():
                    if node.attribute.type == 'c':
                        if key == 0:

                            q.append((f"<= {node.bestsplitc}", child))
                        else:
                            q.append((f"> {node.bestsplitc}", child))
                    else:
                        q.append((key, child))
                counter=0
                for (key, child) in node.childnodes.items():
                    if node.attribute.type != 'c':
                        if child.label!=None:
                            f.write(f"\n{node.id}->{child.label}[label=\"{key}\"];\n")
                        else:
                            f.write(f"\n{node.id}->{child.id}[label=\"{key}\"];\n")
                    else:
                        if key == 0:
                            if child.label!=None:
                                f.write(f"\n{node.id}->{child.label}[label= \"<={node.bestsplitc}\"];\n")
                            else:
                                f.write(f"\n{node.id}->{child.id}[label= \"<={node.bestsplitc}\"];\n")
                        else:
                            if child.label!=None:
                                f.write(f"\n{node.id}->{child.label}[label= \">={node.bestsplitc}\"];\n")
                            else:
                                f.write(f"\n{node.id}->{child.id}[label= \">={node.bestsplitc}\"];\n")

        f.write("\n}")
        f.close()

# Given a training data it returns the attribute list
def generate_attribute_list(training_data):
    d = np.array(training_data)
    value = d.transpose()
    value = value.tolist()

    # Create the list of attribute
    age = Attribute(pid=0, name="age", dtype='c', ntype=0 , values=value[0])
    gender = Attribute(pid=1, name="gender", dtype='d', ntype=2 , values=[0.0,1.0])
    pain = Attribute(pid=2, name="Chest Pain Type", dtype='d', ntype=4 , values=[1.0,2.0,3.0,4.0])
    systolicp = Attribute(pid=3, name="systollicpressure", dtype='c', ntype=0 , values=value[3])
    cholestrol = Attribute(pid=4, name="serum cholestrol", dtype='c', ntype=0 , values=value[4])
    sugar = Attribute(pid=5, name="Fasting Blood Sugar", dtype='d', ntype=2 , values=[1.0,0.0])
    ecg = Attribute(pid=6, name="ECG", dtype='d', ntype=3 , values=[0.0,1.0,2.0])
    maxhr = Attribute(pid=7, name="Maximum Heart Rate", dtype='c', ntype=0 , values=value[7])
    angina = Attribute(pid=8, name="Angina", dtype='d', ntype=2 , values=[0.0,1.0])
    oldpeak = Attribute(pid=9, name="Oldpeak", dtype='c', ntype=0 , values=value[9])
    stpeak = Attribute(pid=10, name="Slope of Peak Exercise ST segment", dtype='d', ntype=3 , values=[1.0,2.0,3.0])
    majorcv = Attribute(pid=11, name="Major Vessels Coloured", dtype='d', ntype=4 , values=[0.0,1.0,2.0,3.0])
    thal = Attribute(pid=12, name="thal", dtype='d', ntype=3 , values=[3.0,6.0,7.0])
    result = Attribute(pid=13, name="Distinct Class Value", dtype='d', ntype=2 , values=[0.0,1.0])

    attributelist = [age, gender, pain, systolicp, cholestrol, sugar, ecg, maxhr, angina, oldpeak, stpeak, majorcv, thal]
    return attributelist


# This function generates a random permutation to build test and training set
def create_training_test_data(data):
    training_set = get_random()     # Random subset of permutation of size 216 (80%)
    training_data = []              # Stores the training data
    test_data = []                  # Stores the test data
    for i in training_set:
        training_data.append(data[i])
    for i in range(ROWS):
        if i not in training_set:
            test_data.append(data[i])

    return (training_data, test_data, training_set)

# 10 random splitting for question - 1 and question - 2
# It also returns the traning set that gives the best accuracy
def random_splitting(data):
    print("Gini based DT accuracy vs. Information Gain based DT accuracy:")
    print("=============================================================")
    avg_gini_acc = 0
    avg_ig_acc = 0
    best_acc = 0
    best_split = None
    for _ in range(RANDOM_SPLITS):
        (training_data, test_data, training_set) = create_training_test_data(data)
        attributelist = generate_attribute_list(training_data)
        # DT based on GINI hill climbing function
        D1 = DecisionTree()
        D1.generateDT(attributelist, training_data, "GINI")
        accuracy1 = D1.accuracy_test(test_data)
        # DT based on Information gain hill climbing function
        D2 = DecisionTree()
        D2.generateDT(attributelist, training_data, "ENTROPY")
        accuracy2 = D2.accuracy_test(test_data)

        print(f"<Gini based DT - {accuracy1}>, <Information Gain based DT - {accuracy2}>")
        avg_gini_acc += accuracy1
        avg_ig_acc += accuracy2

        # Update training set if accuracy is more here
        if max(accuracy1, accuracy2) > best_acc:
            best_acc = max(accuracy1, accuracy2)
            best_split = [training_data, test_data]

    avg_gini_acc /= RANDOM_SPLITS
    avg_ig_acc /= RANDOM_SPLITS
    # Print avg performance of both trees
    print(f"<Avg. accuracy Gini based DT - {avg_gini_acc}>, <Avg. accuracy Information Gain based DT - {avg_ig_acc}>")
    print("==================================================================================================================")

    return best_split

# Plots depth vs. accuracy and total node count vs. accuracy (Q3)
def depth_node_analysis(data):
    for _ in range(RANDOM_SPLITS // 2):
        # Plot for height vs. Accuracy
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set(xlabel = 'Max Depth',
            ylabel = 'Accuracy',
            title = 'Max Depth vs. Accuracy Graph')

        # Plot for count of nodes vs. Accuracy
        # fig2, ax2 = plt.subplots()
        ax2.set(xlabel = 'Count of Nodes',
            ylabel = 'Accuracy',
            title = 'Nodes Count vs. Accuracy Graph')

        Gini_plot = []      # Stores the accuracy of Gini based DT 
        Gini_count = []     # Stores the count of nodes in the corresponding Gini based DT
        IG_plot = []        # Stores the accuracy of IG based DT
        IG_count = []       # Stores the count of nodes in the corresponding IG based DT

        (training_data, test_data, _) = create_training_test_data(data)
        attributelist = generate_attribute_list(training_data)

        for max_depth in range(MAX_DEPTH + 1):
            # DT based on GINI hill climbing function
            D1 = DecisionTree(max_depth)
            D1.generateDT(attributelist, training_data, "GINI")
            accuracy1 = D1.accuracy_test(test_data)
            Gini_count.append(D1.count)
            # DT based on Information gain hill climbing function
            D2 = DecisionTree(max_depth)
            D2.generateDT(attributelist, training_data, "ENTROPY")
            accuracy2 = D2.accuracy_test(test_data)
            IG_count.append(D2.count)
            Gini_plot.append(accuracy1)
            IG_plot.append(accuracy2)


        # Plot the corresponding depth vs. accuracy graph
        x = range(MAX_DEPTH + 1)
        ax1.plot(x, Gini_plot, "oy", label = "Gini Based DT")
        ax1.plot(x, IG_plot, "or", label = "IG Based DT")
        ax1.legend(shadow = True, fancybox = True)
        ax1.plot(x, Gini_plot, 
                x, Gini_plot, "oy",
                x, IG_plot,
                x, IG_plot, "or")

        ax2.plot(Gini_count, Gini_plot, "oy", label = "Gini Based DT")
        ax2.plot(IG_count, IG_plot, "or", label = "IG Based DT")
        ax2.legend(shadow = True, fancybox = True)
        ax2.plot(Gini_count, Gini_plot, 
                Gini_count, Gini_plot, "oy",
                IG_count, IG_plot,
                IG_count, IG_plot, "or")
        
        plt.show()

def depth_node_analysis_best(training_data,test_data):
    for _ in range(1):
        # Plot for height vs. Accuracy
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set(xlabel = 'Max Depth',
            ylabel = 'Accuracy',
            title = 'Max Depth vs. Accuracy Graph')

        # Plot for count of nodes vs. Accuracy
        # fig2, ax2 = plt.subplots()
        ax2.set(xlabel = 'Count of Nodes',
            ylabel = 'Accuracy',
            title = 'Nodes Count vs. Accuracy Graph')

        Gini_plot = []      # Stores the accuracy of Gini based DT 
        Gini_count = []     # Stores the count of nodes in the corresponding Gini based DT
        IG_plot = []        # Stores the accuracy of IG based DT
        IG_count = []       # Stores the count of nodes in the corresponding IG based DT

        # (training_data, test_data, _) = create_training_test_data(data)
        attributelist = generate_attribute_list(training_data)

        for max_depth in range(MAX_DEPTH + 1):
            # DT based on GINI hill climbing function
            D1 = DecisionTree(max_depth)
            D1.generateDT(attributelist, training_data, "GINI")
            accuracy1 = D1.accuracy_test(test_data)
            # accuracy3 = D1.accuracy_test(training_data)
            Gini_count.append(D1.count)
            # DT based on Information gain hill climbing function
            D2 = DecisionTree(max_depth)
            D2.generateDT(attributelist, training_data, "ENTROPY")
            accuracy2 = D2.accuracy_test(test_data)
            IG_count.append(D2.count)
            Gini_plot.append(accuracy1)
            IG_plot.append(accuracy2)


        # Plot the corresponding depth vs. accuracy graph
        x = range(MAX_DEPTH + 1)
        ax1.plot(x, Gini_plot, "oy", label = "Gini Based DT")
        ax1.plot(x, IG_plot, "or", label = "IG Based DT")
        ax1.legend(shadow = True, fancybox = True)
        ax1.plot(x, Gini_plot, 
                x, Gini_plot, "oy",
                x, IG_plot,
                x, IG_plot, "or")

        ax2.plot(Gini_count, Gini_plot, "oy", label = "Gini Based DT")
        ax2.plot(IG_count, IG_plot, "or", label = "IG Based DT")
        ax2.legend(shadow = True, fancybox = True)
        ax2.plot(Gini_count, Gini_plot, 
                Gini_count, Gini_plot, "oy",
                IG_count, IG_plot,
                IG_count, IG_plot, "or")
        
        plt.show()

def main():
    data = []   # Stores the whole dataset
    with open('./heart.dat') as f:
        lines = f.readlines()
    for line in lines:
        data.append([float(x) for x in line.split()])

    # Best Split contains (best training data, corresponding test data, correspoding training set permutation)
    best_split = random_splitting(data)
    # Function call for depth vs. accuracy and node count vs. accuracy analysis
    # depth_node_analysis(data)
    # depth_node_analysis_best(best_split[0],best_split[1])          # uncomment this to see the plots
    # Print Tree
    
    attrib_list = generate_attribute_list(best_split[0])
    D = DecisionTree()
    # D1 = DecisionTree(maxdepth=3)
    D.generateDT(attrib_list, best_split[0], "ENTROPY")
    acc = D.accuracy_test(best_split[1])
    acc2 = D.accuracy_test(best_split[0])
    print(acc, acc2, D.depth,D.count)
    D.createIMG("before")
    D.post_pruning()
    acc = D.accuracy_test(best_split[1])
    acc2 = D.accuracy_test(best_split[0])
    print(acc, acc2, D.depth, D.count)
    # D.createIMG("after")       # Uncomment this to create DOT File
    D.print_tree()            # Uncomment this to print the tree
    

if __name__ == "__main__":
    main()

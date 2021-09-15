import numpy as np
import math

INF = 1000

def prob(v,pos,array):
    total=0
    match=0
    l=len(array)
    for i in range(l):
        if array[i][pos]==v:
            match+=1
    total=l
    return float(match/total)

def get_random():
    import random
    randomlist = random.sample(range(1, 270), 216)
    return randomlist

class Attribute:
    def __init__( self,pid,name, dtype, ntype,values=[] ):
        self.name=name
        self.id= pid
        self.type=dtype # discrete or continuous(d or c)
        self.ntype= ntype   # number of discrete values (0 for c)
        self.values=values  # list of values stored  

    def __str__(self):
        return self.name

class Node:
    def __init__(self):
        self.depth = 0             # depth of node
        self.attribute = None      # attribute of node
        self.pnode = None          # parent node
        self.childnodes = []       # list of doublet lists with childnodes and value for which they are the best fit
        self.lattributes = []      # list of attributes left to use
        self.data = []             # part of data left to be utilized
        self.bestfit = 0           # for continuous values, 0 for discrete values
        self.gini = 0.0
        self.entropy = 0.0

    def createnode(self, attribute, lattributes=[], data=[], bestfit=0, gini=0.0, entropy=0.0):
        a = Node()
        a.depth = self.depth + 1             # depth of node
        a.attribute =  attribute     # attribute of node
        a.childnodes = []       # list of doublet lists with childnodes and value for which they are the best fit
        a.lattributes = lattributes      # list of attributes left to use
        a.data = data             # part of data left to be utilized
        a.bestfit = 0           # for continuous values, 0 for discrete values
        if a.attribute.type == 'c':
            a.bestfit = bestfit
        a.gini = gini
        a.entropy = entropy
        a.pnode = self
        return a

    def bestSplitGini(self,pnode,data, pvalue=None):    # find best split according to gini
        if pnode.attribute in pnode.lattributes:
            attrbs = pnode.lattributes.remove(pnode.attribute)
        else:
            attrbs = pnode.lattributes
        # if attrbs==None or data==[]:
        #     return None
        max = -INF
        bestsplit = pnode.attribute     #just initializing with proper datatype
        bestsplitc = 0
        for attrb in attrbs:
            if attrb.type =='c':
                valueset=[]
                pos = attrb.id
                for i in data:
                    valueset.append(i[pos])
                valueset.sort()
                l = len(valueset)
                bestsplitc = pnode.attribute
                maxc = -INF
                for i in range(l):
                    if i<l-1 and valueset[i]!=valueset[i+1]:
                        p = i+1/l
                        q=1-p
                        j=(valueset[i]+valueset[i+1])/2
                        gini = 0.0
                        datav=[]            # gets dataset with value v for attribute attrb
                        for i in data:
                            pos = attrb.id 
                            if i[pos] < j:
                                datav.append(i)
                        ans = prob(1,13,datav)
                        ans*=(1-ans)
                        p*=ans*2
                        datav=[]
                        for i in data:
                            pos=attrb.id
                            if i[pos]>j:
                                datav.append(i)
                        ans = prob(1,13,datav)
                        ans*=(1-ans)
                        q*=ans*2
                        gini = p+q
                        gain = pnode.gini -gini
                        if maxc<gain:
                            maxc=gain
                            bestsplitc = j
                if max<maxc:
                    max=maxc
                    bestsplit = attrb
            else:       
                gini = 0.0
                for v in attrb.values:
                    a = prob(v, attrb.id, data)
                    datav=[]            # gets dataset with value v for attribute attrb
                    for i in data:
                        pos = attrb.id 
                        if i[pos] == v:
                            datav.append(i)
                    ans = prob(1, 13, datav)
                    ans*=(1-ans)
                    a*=ans*2
                    gini+=a      # coz 2 target values
                ginigain = pnode.gini - gini
                if max <ginigain:
                    max = ginigain
                    bestsplit = attrb
        k=0
        if bestsplit.type=='c':
            k=bestsplitc
        child = pnode.createnode(attribute=bestsplit,lattributes= attrbs, data=data,bestfit=k,gini = max)
        if pvalue !=None:
            branch=[pvalue, child]
            self.childnodes.append(branch)
        return child

    def bestSplitEntropy(self,pnode,data, pvalue=None):            # find best split according to entropy
        if pnode.attribute in pnode.lattributes:
            attrbs = pnode.lattributes.remove(pnode.attribute)
        else:
            attrbs = pnode.lattributes
        max = -INF
        bestsplit = pnode.attribute
        bestsplitc = 0
        for attrb in attrbs:
            if attrb.type == 'c':
                valueset=[]
                pos = attrb.id
                for i in data:
                    valueset.append(i[pos])
                valueset.sort()
                l = len(valueset)
                bestsplitc = pnode.attribute
                maxc = -INF
                for i in range(l):
                    if i<l-1 and valueset[i]!=valueset[i+1]:
                        a = i+1/l           # value of |Sv|/|S|
                        b = 1-a            # value of |Sv|/|S|
                        j=(valueset[i]+valueset[i+1])/2
                        entropysum = 0.0
                        datav=[]            
                        for i in data:
                            pos = attrb.id 
                            if i[pos] < j:
                                datav.append(i)
                        p = prob(1,13,datav)
                        q = 1-p
                        if p != 0.0 and p != 1.0:
                            entropy1 = -p*math.log2(p) - q*math.log2(q)
                        else:
                            entropy1 = 0.0
                        datav=[]
                        for i in data:
                            pos = attrb.id
                            if i[pos] > j:
                                datav.append(i)
                        p = prob(1, 13, datav)
                        q = 1-p
                        if p != 0.0 and p != 1.0:
                            entropy2 = -p*math.log2(p) - q*math.log2(q)
                        else:
                            entropy2 = 0.0
                        entropysum = a*entropy1 + b*entropy2
                        infogain = pnode.entropy- entropysum
                        if maxc<infogain:
                            maxc=infogain
                            bestsplitc = j
            if max<maxc:
                max=maxc
                bestsplit = attrb

            else:
                entropysum=0.0
                for v in attrb.values:
                    a = prob(v, attrb.id, data)     # | Sv|/| S|
                    datav=[]            # gets dataset with value v for attribute attrb
                    for i in data:
                        pos = attrb.id 
                        if i[pos] == v:
                            datav.append(i)
                    p = prob(1, 13, datav)
                    q=1-p
                    entropy = 0.0
                    if p != 0.0 and p != 1.0:
                        entropy = -p*math.log2(p) - q*math.log2(q)
                    else:
                        entropy = 0.0
                    entropysum += a*entropy
                gain = pnode.entropy - entropysum
                if max < gain:
                    max = gain
                    bestsplit = attrb
            k=0
            if bestsplit.type=='c':
                k=bestsplitc

        child = pnode.createnode(attribute=bestsplit,lattributes= attrbs, data=data,bestfit=k,entropy = max  )
        if pvalue !=None:
            branch=[pvalue, child]
            self.childnodes.append(branch)
        return child

    def __str__(self):
        return self.attribute.name
    

class DecisionTree:
    def __init__(self, maxdepth):
        self.maxdepth = maxdepth
        self.nnode = 0
        self.currentdepth = 0
        self.root = Node()

    def makeroot(self, attributelist,data, method="GINI" ):
        root = Node()
        root.depth = 0
        root.data = data
        root.lattributes = attributelist
        if method=="ENTROPY":
            root = root.bestSplitEntropy(root, data)
        else:
            root = root.bestSplitGini(root, data)
        return root

    def growtree(self,node, method="GINI"):
        if node.depth >= self.maxdepth:
            return
        attribute = node.attribute

        if attribute.type=='c':
            data=[]
            pos = attribute.id
            for row in node.data:
                if row[pos]<=node.bestfit:   # for continuous values less than bestfit
                    data.append(row)
            if method=="GINI":
                childnode = node.bestSplitGini(node, data, node.bestfit)
                print(childnode)
                if childnode is None:
                        return
                self.nnode+=1
                if(self.currentdepth<childnode.depth):
                    self.currentdepth=childnode.depth
                growtree(childnode,"GINI")
            else:
                childnode = node.bestSplitEntropy(node, data, node.bestfit)
                print(childnode)
                if childnode is None:
                        return
                self.nnode+=1
                if(self.currentdepth<childnode.depth):
                    self.currentdepth=childnode.depth
                growtree(childnode,"ENTROPY")
            data=[]
            pos = attribute.id
            for row in node.data:
                if row[pos]>node.bestfit:       # for continuous values more than bestfit
                    data.append(row)
            if method=="GINI":
                childnode = node.bestSplitGini(node, data, value)
                print(childnode)
                if childnode is None:
                        return
                self.nnode+=1
                if(self.currentdepth<childnode.depth):
                    self.currentdepth=childnode.depth
                growtree(childnode,"GINI")
            else:
                childnode = node.bestSplitEntropy(node, data, value)
                print(childnode)
                if childnode is None:
                        return
                self.nnode+=1
                if(self.currentdepth<childnode.depth):
                    self.currentdepth=childnode.depth
                growtree(childnode,"ENTROPY")
        
        else:    
            for value in atribute.values:
                data=[]
                pos = attribute.id
                for row in node.data:
                    if row[pos]==value:
                        data.append(row)
                if method=="GINI":
                    childnode = node.bestSplitGini(node, data, value)
                    if childnode is None:
                        return
                    self.nnode+=1
                    if(self.currentdepth<childnode.depth):
                        self.currentdepth=childnode.depth
                    growtree(childnode,"GINI")
                else:
                    childnode = node.bestSplitEntropy(node, data, value)
                    if childnode is None:
                        return
                    self.nnode+=1
                    if(self.currentdepth<childnode.depth):
                        self.currentdepth=childnode.depth
                    growtree(childnode,"ENTROPY")

        return


def printInfo(node,value, width=4,method="GINI"):
    const = int(node.depth *width**1.5)
    spaces = "-" * const
    print(f"|{spaces}  {node}")
    if node.pnode==None:
        print(f"{' ' * const}   | At Value: {value} of {node.pnode}")
    if method=="GINI":
        print(f"{' ' * const}   | GINI impurity of the node: {round(node.gini, 2)}")
    else:
        print(f"{' ' * const}   | Entropy impurity of the node: {round(node.entropy, 2)}")


def printTree(nodebranch, method="GINI"): #pass [root,0] for printing
    printInfo(node=nodebranch[1],value=nodebranch[0],method=method )
    for childbranch in nodebranch[1].childnodes:
        printTree(childbranch, method)
    return




def main():
    training_set=get_random()
    with open('../heart.dat') as f:
        lines = f.readlines()
    data=[]
    for i in training_set:
        data.append(lines[i].split())
    for i in range(216):
        for j in range(14):
            data[i][j]=float(data[i][j])
    d=np.array(data)
    value=d.transpose()
    value=value.tolist()

    age=Attribute(pid=0,name="age", dtype='c', ntype=0 , values=value[0])
    gender=Attribute(pid=1,name="gender", dtype='d', ntype=2 , values=[0.0,1.0])
    pain=Attribute(pid=2,name="Chest Pain Type", dtype='d', ntype=4 , values=[1.0,2.0,3.0,4.0])
    systolicp=Attribute(pid=3,name="systollicpressure", dtype='c', ntype=0 , values=value[3])
    cholestrol=Attribute(pid=4,name="serum cholestrol", dtype='c', ntype=0 , values=value[4])
    sugar=Attribute(pid=5,name="Fasting Blood Sugar", dtype='d', ntype=2 , values=[1.0,0.0])
    ecg=Attribute(pid=6,name="ECG", dtype='d', ntype=3 , values=[0.0,1.0,2.0])
    maxhr=Attribute(pid=7,name="Maximum Heart Rate", dtype='c', ntype=0 , values=value[7])
    angina=Attribute(pid=8,name="Angina", dtype='d', ntype=2 , values=[0.0,1.0])
    oldpeak=Attribute(pid=9,name="Oldpeak", dtype='c', ntype=0 , values=value[9])
    stpeak=Attribute(pid=10,name="Slope of Peak Exercise ST segment", dtype='d', ntype=3 , values=[1.0,2.0,3.0])
    majorcv=Attribute(pid=11,name="Major Vessels Coloured", dtype='d', ntype=4 , values=[0.0,1.0,2.0,3.0])
    thal=Attribute(pid=12,name="thal", dtype='d', ntype=3 , values=[3.0,6.0,7.0])
    result=Attribute(pid=13,name="Distinct Class Value", dtype='d', ntype=2 , values=[0.0,1.0])

    attributelist=[age,gender,pain, systolicp,cholestrol,sugar,ecg,maxhr,angina,oldpeak,stpeak,majorcv,thal]
    D = DecisionTree(8)
    root = D.makeroot(attributelist,data, "ENTROPY")
    D.growtree(root)
    printTree([0,root],"ENTROPY")
    # print(root)
    

    

if __name__ == "__main__":
    main()
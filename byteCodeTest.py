import dgl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import pydot
import os
import collections
import random
from model_test import HAN
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import torch
import json
def creatNodeLabel(g,d):
    nodes = g.nodes()
    int2label = {}
    for idx,node_idx in enumerate(nodes):
        obj = g._node[node_idx]
        if 'shape' not in obj:
            shape = 'None'
        else:
            shape = obj['shape']
        if 'color' not in obj:
            color = 'None'
        else:
            color = obj['color']
        if 'fillcolor' not in obj:
            fillcolor = 'None'
        else:
            fillcolor = obj['fillcolor']
        t = (shape,color,fillcolor)
        node_type = d[t]
        # nodetype without number
        # psb_node_type = g._node[node_idx]['label'].splitlines()[0]
        # colonPos = psb_node_type.find(':') + 1
        # node_type = psb_node_type[colonPos:]
        int2label[node_idx] = node_type
        #print(node_idx,node_type)

    return int2label

def createdgelabel(g):
    edgeLabel = {}
    edgedata = g.edges.data()
    for u,v,fe in edgedata:
        edgeLabel[(u,v)] = 'CF'

    return edgeLabel

def graphVision(g,d):
    nodesLabel = creatNodeLabel(g,d)
    edgeLabel = createdgelabel(g)

    pos = nx.spring_layout(g)

    nx.draw_networkx_labels(g, pos, nodesLabel, font_size=6, font_color="black")
    nx.draw_networkx_edge_labels(g,pos,edgeLabel,font_size=6,font_color="black")
    nx.draw(g,pos)
    plt.show()


def createHeteroGraph(g):
    candiGraphData = {}
    graphdata = {}
    nodesLabel = creatNodeLabel(g)
    u_vector, v_vector = 0, 1
    for u, v, la in g.edges.data():
        if la:
            e_type = la['label']
        else:
            e_type = 'CF'
        if (nodesLabel[u],e_type,nodesLabel[v]) not in candiGraphData:
            candiGraphData[(nodesLabel[u],e_type,nodesLabel[v])] = [[int(u)],[int(v)]]
        else:
            candiGraphData[(nodesLabel[u],e_type,nodesLabel[v])][u_vector].append(int(u))
            candiGraphData[(nodesLabel[u],e_type,nodesLabel[v])][v_vector].append(int(v))
    for key in candiGraphData:
        u = candiGraphData[key][u_vector]
        v = candiGraphData[key][v_vector]
        graphdata[key] = (th.tensor(u),th.tensor(v))
    dg = dgl.heterograph(graphdata)
    return dg

#return a dg which reflect itself
def reflectHeteroGraph(g,d):
    candiGraphData = {}
    graphdata = {}
    nodesLabel = creatNodeLabel(g,d)
    u_vector, v_vector = 0, 1
    for u,v,la in g.edges.data():
        if la:
            e_type = la['label']
        else:
            e_type = 'CF'
        if (nodesLabel[u],e_type,nodesLabel[v]) not in candiGraphData:
            candiGraphData[(nodesLabel[u],e_type,nodesLabel[v])] = [[int(u)],[int(v)]]
        else:
            candiGraphData[(nodesLabel[u],e_type,nodesLabel[v])][u_vector].append(int(u))
            candiGraphData[(nodesLabel[u],e_type,nodesLabel[v])][v_vector].append(int(v))
    for key in candiGraphData:
        u = candiGraphData[key][u_vector]
        v = candiGraphData[key][v_vector]
        graphdata[key] = (th.tensor(u),th.tensor(v))
        sn,e,en = key
        new_e = e[::-1]
        new_k = (en,new_e,sn)
        graphdata[new_k] = (th.tensor(v),th.tensor(u))
    dg = dgl.heterograph(graphdata)

    return dg

#read all the file name in filepath, return a map where smartcontract name mapped to the file name.
# def travelsalDir(filepath):
#     data=os.listdir(filepath)
#     list_data = {}
#     for name in data:
#         if '.' not in name:
#             if 'cleanset' in name.split('_'):
#                 list_data[0] = name
#             else:
#                 list_data[1] = name
#     pathDict = {}
#     labels = collections.defaultdict(int)
#     name_list = []
#     # dir is the vulnerabilities name
#     for l in list_data:
#         files = os.listdir(filepath+list_data[l])
#         for file in files:
#             end = file.find('.')
#             pathDict[file[:end]] = (filepath + list_data[l] + '/' + file)
#             labels[file[:end]] = th.LongTensor([l])
#             name_list.append(file[:end])
#
#     return pathDict, labels,name_list

def travelsalDir(filepath):
    count_0, count_1 = 0, 0
    with open(filepath,'r') as f:
        load_dict = json.load(f)
    location = '/Users/xiechunyao/bytecode_cfg_set/'

    pathDict = {}
    labels = collections.defaultdict(int)
    name_list = []
    for name in load_dict:
        if load_dict[name] == 0 and count_0 < 100:
            count_0 += 1
            labels[name] = th.LongTensor([load_dict[name]])
            pathDict[name] = location + name + '.dot'
            name_list.append(name)
        elif load_dict[name] == 1 and count_1 < 100:
            count_1 += 1
            labels[name] = th.LongTensor([load_dict[name]])
            pathDict[name] = location + name + '.dot'
            name_list.append(name)

    return pathDict, labels, name_list


#create dg set,return a dict where smartcontractname map to a set of cfgs
def createGraph(pathDict,d):
    graphDict = {}
    for file in pathDict:
        print(file)
        nx_g = nx.drawing.nx_pydot.read_dot(pathDict[file])

        if len(nx_g.edges) > 0:
            #dg = createHeteroGraph(nx_g)
            dg = reflectHeteroGraph(nx_g,d)
            graphDict[file] = dg

    return graphDict


def graphFeature(graph_dict):
    nodeSet = set()
    edgeSet = set()
    for k in graph_dict:
        g = graph_dict[k]
        for n in g.ntypes:
            nodeSet.add(n)
        for e in g.etypes:
            edgeSet.add(e)
    nodeList = list(nodeSet)
    n = len(nodeList) # nums of node types
    node2oneHot = {}
    for i, node in enumerate(nodeList):
        node2oneHot[node] = [0 for _ in range(n)]
        node2oneHot[node][i] = 1
    return node2oneHot, n

def feature4dg(dg,node2oneHot):
    for node in dg.ntypes:
        n = dg.num_nodes(node)
        l = [node2oneHot[node] for _ in range(n)]
        dg.nodes[node].data['f'] = th.tensor(l).float()
    return dg

def feature4allgraph(graph_dict,node2oneHot):
    #contract_list = []
    for k in graph_dict:
        #contract_list.append(k)
        dg = graph_dict[k]
        feature4dg(dg,node2oneHot)
    return graph_dict

d = {('None', 'None', 'None'): '0', ('None', 'None', 'orange'): '1', ('Msquare', 'None', 'gold'): '2', ('None', 'None', 'lemonchiffon'): '3', ('Msquare', 'crimson', 'crimson'): '4', ('None', 'None', 'crimson'): '5', ('Msquare', 'crimson', 'None'): '6', ('Msquare', 'crimson', 'lemonchiffon'): '7'}

# filepath = '/Users/xiechunyao/Downloads/bytecode_dataset/'
filepath = '/Users/xiechunyao/labels.json'

pathDict, labels, name_list = travelsalDir(filepath)
graphDICT = createGraph(pathDict,d)
node2oneHot, numsofnodetype = graphFeature(graphDICT)
f_graphDICT= feature4allgraph(graphDICT,node2oneHot)
random.shuffle(name_list)
metapaths = [['CF', 'FC']]
input_size = numsofnodetype
data = []
for con in name_list:
    g = f_graphDICT[con]
    l = labels[con]
    data.append((g,l))
data = np.array(data)

model = HAN(meta_paths=metapaths,
            in_size=8,
            hidden_size=64,
            out_size=2,
            num_heads=[8],
            dropout=0)
loss_fcn = th.nn.CrossEntropyLoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001,
                          weight_decay=0.001)
model.train()
def evaluate2(data,m):
    y_true = []
    y_pred = []
    for g,l in data:
        logit = m(g,g.ndata['f'])
        predict = torch.max(logit,1)[1]
        y_pred.append(predict)
        y_true.append(l)
    print(y_true)
    print(y_pred)
    acc = accuracy_score(y_true, y_pred)
    micro = metrics.precision_score(y_true, y_pred, average='micro')
    macro = metrics.precision_score(y_true, y_pred, average='macro')
    recall_micro = metrics.recall_score(y_true, y_pred, average='micro')
    recall_macro = metrics.recall_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    d = {'acc': acc,'micro': micro, 'macro': macro, 'recall_micro': recall_micro, 'recall_macro': recall_macro, 'f1': f1}
    return d
# def evaluate2(data,m):
#     y_true = []
#     y_pred = []
#     for g,l in data:
#         features = g.ndata['f']
#         logit = m(g,features)
#         predict = th.max(logit,1)[1]
#         y_pred.append(predict)
#         y_true.append(l)
#         print(predict,l)
#     acc = accuracy_score(y_true, y_pred)
#     print(acc)
#     return acc
matrix = []
kf = KFold(n_splits=5,random_state=1, shuffle=True)
cnt = 0
for train_idx, test_idx in kf.split(data):
    train_set = data[train_idx]
    test_set = data[test_idx]

    for epoch in range(10):
        #model.train()
        #print('this is ',epoch,'epoch')
        #model.train()
        for g,l in train_set:

            features = g.ndata['f']

            logits = model(g,features)
            loss = loss_fcn(logits, l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch,loss.item())
    cnt += 1
    scores = evaluate2(test_set,model)
    matrix.append(scores)
    print('this is', cnt, 'time mat:',scores)

acc = []
micro = []
macro = []
recall_micro = []
recall_macro = []
f1 = []

for obj in matrix:
    acc.append(obj['acc'])
    micro.append(obj['micro'])
    macro.append(obj['macro'])
    recall_micro.append(obj['recall_micro'])
    recall_macro.append(obj['recall_macro'])
    f1.append(obj['f1'])
print('acc is:', acc)
print('micro is:', micro)
print('macro is:', macro)
print('recall_micro is:', recall_micro)
print('recall_macro is:', recall_macro)
print('f1 is:', f1)

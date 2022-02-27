import json
import os
import re
import time
import numpy as np
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch as th
import vec2onehot
import graph2vec
import json
from model_dgl import HeteroClassifier
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from model_test import HAN

def create_graph_data(node_feature, edge_feature):
    node_encode, var_encode, node_embedding, var_embedding = graph2vec.embedding_node(node_feature)
    edge_encode, edge_embedding = graph2vec.embedding_edge(edge_feature)
    #print('edge_encode',node_embedding)

    node_d = {}
    edge_d = {}
    edges_type = {'AG':'DF', 'AC':'DF', 'FB':'FB','FW':'FW'} # DF is data flow, FB is fall back, FW is forward, CF is control flow
    graph_data = {}
    edgeAddfeature = {'NULL':[0,0,0,0,0],'DF':[0,1,0,0,0],'FB':[0,0,1,0,0],'FW':[0,0,0,1,0],'CF':[0,0,0,0,1]}
    node_list = {'M':[],'S':[],'F':[]}
    main_node_id = 0
    s_node_id = 0
    f_node_id = 0
    node_embedding_dict = {}
    node_feature2dict = {'M':[],'S':[],'F':[]}
    edge_feature2dict = {}
    for node in node_embedding:
        node_embedding_dict[node[0]] = node[1]
    for var in var_embedding:
        node_embedding_dict[var[0]] = var[1]

    for node in node_feature:
        if node[0][0] in ['C', 'W', 'S']: # if 'F' as else nodes use this#if node[0][0] in ['C', 'W', 'S', 'F']:
            node_feature2dict['M'].append(node_embedding_dict[node[0]].tolist())
            node_d[node[0]] = ['M',main_node_id]
            node_list['M'].append('M'+str(main_node_id))

            main_node_id += 1

        elif node[0][0] == 'V':
            node_feature2dict['S'].append(node_embedding_dict[node[0]].tolist())
            node_d[node[0]] = ['S',s_node_id]
            node_list['S'].append('S'+str(s_node_id))
            s_node_id += 1

        else:
            node_feature2dict['F'].append(node_embedding_dict[node[0]].tolist())
            node_d[node[0]] = ['F',f_node_id]
            node_list['F'].append('F'+str(f_node_id))
            f_node_id += 1


    for i in range(len(edge_feature)):
        edge = edge_feature[i]
        candidateEembedding = edge_embedding[i][2].tolist()[:-12]

        if edge[4] in edges_type.keys():
            edge_type = edge[4]
            candidateEembedding += edgeAddfeature[edge_type]
        else:
            edge_type = 'CF'
            candidateEembedding += edgeAddfeature[edge_type]
        edge_embedding[i][2].tolist()
        sn_type, dn_type = (node_d[edge[0]][0], node_d[edge[1]][0])
        sn_id, dn_id = (node_d[edge[0]][1], node_d[edge[1]][1])
        graph_data_k = (sn_type, edge_type, dn_type)

        edge_k = edge_type

        if edge_k not in edge_d.keys():
            edge_d[edge_k] = [(sn_type+str(sn_id), dn_type+str(dn_id))]
        else:
            edge_d[edge_k].append((sn_type+str(sn_id), dn_type+str(dn_id)))

        if graph_data_k not in graph_data.keys():
            graph_data[graph_data_k] = [[sn_id],[dn_id]]
            edge_feature2dict[graph_data_k] = [candidateEembedding]
        else:
            graph_data[graph_data_k][0].append(sn_id)
            graph_data[graph_data_k][1].append(dn_id)
            edge_feature2dict[graph_data_k].append(candidateEembedding)
    return graph_data, edge_d, node_list, node_feature2dict,edge_feature2dict #edge_d = {edge_type:[[startnode1, endnode1],[startnode2,endnode2]]}

def dgl_graph(graph_data):
    keys = []
    for key in graph_data.keys():
        keys.append(key)

    for key in keys:
        u = th.tensor(graph_data[key][0])
        v = th.tensor(graph_data[key][1])
        graph_data[key] = (u,v)
        s,ed,e = key
        new_t = (e, ed[1]+ed[0],s)
        graph_data[new_t] = (v,u)

    g = dgl.heterograph(graph_data)
    # for key in graph_data.keys():
    #     u = th.tensor(graph_data[key][0])
    #     v = th.tensor(graph_data[key][1])
    #     graph_data[key] = (u,v)
    #
    # g = dgl.heterograph(graph_data)

    return g
#def featuredDglGraph(g,node_feature2dict,edge_feature2dict):
def featuredDglGraph(g,node_feature2dict):
    n_types = g.ntypes
    #e_types = g.canonical_etype()
    for t in n_types:
        n_feature = th.tensor(node_feature2dict[t])
        convert_n_feature = n_feature.float()
        g.nodes[t].data['nfeature'] = convert_n_feature #th.from_numpy(np.array(node_feature2dict[t],dtype=np.double))
        g.nodes[t].data['hv'] = th.randn(g.num_nodes(t),57)
    #for t2 in e_types:
    #g.edges[t2].data['efeature'] = th.from_numpy(np.array(edge_feature2dict[t2]))
    return g

def graph_vision(node_list, edge_d):
    new_node_d = {}
    label2number = {'M':[], 'S':[],'F':[]}
    i = 0
    new_node_list = []
    n_color = {'M':"red", 'S':"green", 'F':"grey"}
    e_color = {'MM':"yellow",'SS':"green", 'MS':"blue", 'SM':"black"}
    for k in node_list.keys():
        label2number[k] = []
        for node in node_list[k]:
            new_node_d[node] = i
            label2number[k].append(i)
            new_node_list.append(node)
            i += 1
    n = len(new_node_list)
    G = nx.random_k_out_graph(n,3,0.5)
    pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes

    # nodes
    options = {"edgecolors": "tab:gray", "node_size": 1000, "alpha": 0.9}
    for key in label2number:
        nx.draw_networkx_nodes(G, pos, nodelist=label2number[key], node_color=[n_color[key]] * len(label2number[key]),**options)

    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    for key in edge_d.keys():
        edgelist = []
        for t0,t1 in edge_d[key]:
            edgelist.append((new_node_d[t0],new_node_d[t1]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            width=8,
            alpha=0.5,
            edge_color=[e_color[key]] * len(edgelist)
        )

    # some math labels

    labels = {}
    for i in range(n):
        labels[i] = new_node_list[i]
    nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color="whitesmoke")
    plt.tight_layout()
    plt.axis("off")
    plt.show()

def graph_vision2(node_list, edge_d):
    n_color = {'M':"yellow", 'S':"green", 'F':"red"}
    e_feature = {'CF':["red",1], 'DF':["blue",1], 'FW':["black",1],'FB':["yellow",1]}
    dg = nx.DiGraph()
    node_color = []
    edge_labels = {}
    for k in node_list.keys():
        for node in node_list[k]:
            dg.add_node(node)
            node_color.append(n_color[k])
    for k in edge_d.keys():
        for t0,t1 in edge_d[k]:
            dg.add_edge(t0,t1,color=e_feature[k][0],weight=e_feature[k][1])
            edge_labels[(t0,t1)] = k
    edges = dg.edges()
    edge_color = [dg[u][v]['color'] for u,v in edges]
    weights = [dg[u][v]['weight'] for u,v in edges]

    pos = nx.spring_layout(dg,seed=3)

    nx.draw(dg,pos,node_color=node_color,edge_color=edge_color,width=weights,with_labels=True)
    nx.draw_networkx_edge_labels(dg,pos,edge_labels=edge_labels,font_color='black')
    plt.show()

def dataloader(feature_data, target_data):
    labels = []
    graphs = []
    features = []
    cnt = 0
    with open(feature_data, 'r') as rf1:
        data1 = json.load(rf1)
    with open(target_data, 'r') as rf2:
        data2 = json.load(rf2)

    for i,item in enumerate(data2):
        contract_name = item['contract_name']
        if contract_name in data1:
            node_feature = data1[contract_name]['node_feature']
            edge_feature = data1[contract_name]['edge_feature']
            graph_data, edge_d, node_d,node_feature2dict, edge_feature2dict = create_graph_data(node_feature, edge_feature)
            g = dgl_graph(graph_data)
            featured_g = featuredDglGraph(g,node_feature2dict)
            label = int(item['targets'])
            if label == 1:
                cnt += 1
            labels.append(th.LongTensor([label]))
            graphs.append(featured_g)
            # hg = 0
            # for ntype in g.ntypes:
            #     hg = hg + dgl.mean_nodes(g, 'nfeature', ntype=ntype)
            # features.append(hg)

    return labels,graphs,cnt

f1 = 'graph_data.json'
f2 = 'Reentrancy_AutoExtract_fullnodes.json'
labels,graphs,cnt= dataloader(f1,f2)
train_graphs = graphs[:120]
test_graphs = graphs[120:]
train_l = labels[:120]
test_l = labels[120:]

data = np.array([(graphs[i],labels[i]) for i in range(len(graphs))])
train_data = np.array([(train_graphs[i],train_l[i]) for i in range(len(train_graphs))])
test_data = np.array([(test_graphs[i],test_l[i]) for i in range(len(test_graphs))])
X_data = np.array([graphs[i] for i in range(len(graphs))])
y_data = np.array([labels[j] for j in range(len(labels))])

meta_paths = [['CF', 'FC'],['DF', 'FD'], ['FB', 'BF'], ['FW','WF']]
model = HAN(meta_paths=meta_paths,
            in_size=57,
            hidden_size=32,
            out_size=2,
            num_heads=[8],
            dropout=0.1)


loss_fcn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005,
                             weight_decay=0.001)
# def evaluate(data,m):
#     i = 0
#     j = 0
#     for g,l in data:
#         features = g.ndata['nfeature']
#         logit = m(g,features)
#
#         predict = torch.max(logit,1)[1]
#         print(logit,predict,l)
#         if predict == l:
#             i += 1
#         j += 1
#         acc = i / j
#     print(acc)
#     return acc
def evaluate2(data,m):
    y_true = []
    y_pred = []
    for g,l in data:
        features = g.ndata['nfeature']
        logit = m(g,features)
        predict = torch.max(logit,1)[1]
        y_pred.append(predict)
        y_true.append(l)
        print(predict,l)
    acc = accuracy_score(y_true, y_pred)
    print(acc)
    return acc
kf = KFold(n_splits=5)
result = []
for train_idx, test_idx in kf.split(data):
    train_set = data[train_idx]
    test_set = data[test_idx]
    for epoch in range(20):
        print('this is ',epoch,'epoch')
        model.train()
        for g,l in train_set:

            features = g.ndata['nfeature']

            logits = model(g,features)
            loss = loss_fcn(logits, l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    acc = evaluate2(test_data,model)
    result.append(acc)
print(result)


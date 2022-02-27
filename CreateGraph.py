import networkx as nx
import dgl
import torch as th


def creatDGLGraph(file):

    # d is a key, value map, where key = (blockshape, rim_color, fillcolor) is a tupel of node attributes
    d = {('None', 'None', 'None'): '0', ('None', 'None', 'orange'): '1', ('Msquare', 'None', 'gold'): '2', ('None', 'None', 'lemonchiffon'): '3', ('Msquare', 'crimson', 'crimson'): '4', ('None', 'None', 'crimson'): '5', ('Msquare', 'crimson', 'None'): '6', ('Msquare', 'crimson', 'lemonchiffon'): '7'}

    g = nx.drawing.nx_pydot.read_dot(file)

    nodes = g.nodes()
    int2label = {}
    # to determine the node type of every block in the nx graph
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

        int2label[node_idx] = node_type

        # to convert nx graph into dgl graph structure
        candiGraphData = {}
        graphdata = {}
        nodesLabel = int2label
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

path = '.../.../file.dot'
creatDGLGraph(path)
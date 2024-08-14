import networkx as nx
import matplotlib.pyplot as plt
from decimal import Decimal

class Application:
    def __init__(self, aID, pList):
        self.aID=aID
        self.processList=pList

        
    def printApplication(self):
        print( '{}\t:  {}'.format(self.aID,self.processList))
        
    def __repr__(self):
        return str(self.aID)

class Process:
    def __init__(self, pID, app, nOPS, freq):
        self.pID=pID
        self.nOPS=nOPS
        self.freq=freq
        self.ins=[]
        self.outs=[]
        self.inBW=0
        self.outBW=0
        self.app=app
    
    def addMessage(self,m):
        if(m.source==self.pID):
            self.outs.append(m.mID)
            self.outBW+=m.size*m.freq
            
        if(m.dest==self.pID):
            self.ins.append(m.mID)
            self.inBW+=m.size*m.freq
           
    def printProcess(self):
        print( '{}({})\t: {:.2E}OPS, {}Hz, i:{:.2E}bps o:{:.2E}bps'.format(self.pID,self.app,self.nOPS,self.freq,self.inBW,self.outBW,))
    
    def __repr__(self):
        return str(self.pID)
    
    #def __str__(self):
        #return str(self.pID)
    
class Message:
    def __init__(self, mID, source, dest, freq, size):
        self.mID=mID
        self.source=source
        self.dest=dest
        self.freq=freq
        self.size=size
        
    def printMessage(self):
        print( '{}\t: {}->{}:{}b {}Hz'.format(self.mID,self.source,self.dest,self.size,self.freq))


class AAM:

    def __init__(self,appList,processList, messageList):

        self.appList=appList
        self.processList=processList
        self.messageList=messageList

        self.ProcessGraph=nx.Graph()

        for value in processList.values():
            self.ProcessGraph.add_node(value)   

        for value in messageList.values():   
            self.ProcessGraph.add_edge(processList[str(value.source)],processList[str(value.dest)])

        self.AppGraph=nx.Graph()

        for value in appList.values():
            self.AppGraph.add_node(value)

        for value in messageList.values():
            self.AppGraph.add_edge(appList[str(processList[str(value.source)].app)],appList[str(processList[str(value.dest)].app)])



    def drawAAMGraphs(self):
        
        graphs=[self.AppGraph,self.ProcessGraph]
        

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
        ax = axes.flatten()
        pos=[]
        titles=["Application Graph","Process Graph"]
        
        for i in range(len(graphs)): 
            pos.append(nx.spring_layout(graphs[i], k= 2.0, weight=2,scale=10,iterations=50))
        
        for i in range(len(graphs)): 

            ax[i].set_title(titles[i])
            ax[i].set_axis_off()

            nx.draw_networkx_nodes(graphs[i] ,pos[i],ax=ax[i])
            nx.draw_networkx_labels(graphs[i] ,pos[i], font_size=10,ax=ax[i])
            nx.draw_networkx_edges(graphs[i] ,pos[i], width=1.0, alpha=0.5,ax=ax[i])
  
            nx.draw_networkx_nodes(graphs[i] ,pos[i],ax=ax[i])
            nx.draw_networkx_labels(graphs[i] ,pos[i], font_size=10,ax=ax[i])
            nx.draw_networkx_edges(graphs[i] ,pos[i], width=1.0, alpha=0.5,ax=ax[i])
            
        plt.show()
        





   

        
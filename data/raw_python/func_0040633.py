def plotGraph(self,mode="plain",nodes=None,filename="tgraph.png"):
        """Plot graph with nodes (iterable) into filename
        """
        if nodes==None:
            nodes=self.nodes
        else:
            nodes=[i for i in self.nodes if i in nodes]
        for node in self.nodes:
            n_=self.A.get_node(node)
            if mode=="plain":
                nmode=1
            else:
                nmode=-1
            pos="{},{}".format(self.xi[::nmode][self.nm.nodes_.index(node)],self.yi[::nmode][self.nm.nodes_.index(node)])
            n_.attr["pos"]=pos
            n_.attr["pin"]=True
            color='#%02x%02x%02x' % tuple([255*i for i in self.cm[int(self.clustering[n_]*255)][:-1]])
            n_.attr['fillcolor']= color
            n_.attr['fixedsize']=True
            n_.attr['width']=  abs(.1*(self.nm.degrees[n_]+  .5))
            n_.attr['height']= abs(.1*(self.nm.degrees[n_]+.5))
            n_.attr["label"]=""
            if node not in nodes:
                n_.attr["style"]="invis"
            else:
                n_.attr["style"]="filled"
        for e in self.edges:
            e.attr['penwidth']=3.4
            e.attr["arrowsize"]=1.5
            e.attr["arrowhead"]="lteeoldiamond"
            e.attr["style"]=""
            if sum([i in nodes for i in (e[0],e[1])])==2:
                e.attr["style"]=""
            else:
                e.attr["style"]="invis"
        tname="{}{}".format(self.basedir,filename)
        print(tname)
        self.A.draw(tname,prog="neato")
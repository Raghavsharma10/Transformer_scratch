def init_all(self,roots=None,inverted_edges=None,optimize=False):
        """initializes the layout algorithm by computing roots (unless provided),
           inverted edges (unless provided), vertices ranks and creates all dummy
           vertices and layers. 
             
             Parameters:
                roots (list[Vertex]): set *root* vertices (layer 0)
                inverted_edges (list[Edge]): set edges to invert to have a DAG.
                optimize (bool): optimize ranking if True (default False)
        """
        if self.initdone: return
        # For layered sugiyama algorithm, the input graph must be acyclic,
        # so we must provide a list of root nodes and a list of inverted edges.
        if roots==None:
            roots = [v for v in self.g.sV if len(v.e_in())==0]
        if inverted_edges==None:
            L = self.g.get_scs_with_feedback(roots)
            inverted_edges = [x for x in self.g.sE if x.feedback]
        self.alt_e = inverted_edges
        # assign rank to all vertices:
        self.rank_all(roots,optimize)
        # add dummy vertex/edge for 'long' edges:
        for e in self.g.E():
            self.setdummies(e)
        # precompute some layers values:
        for l in self.layers: l.setup(self)
        self.initdone = True
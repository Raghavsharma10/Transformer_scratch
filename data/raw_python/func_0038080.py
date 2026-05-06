def to_funset(self):
        """
        Converts the hypergraph to a set of `gringo.Fun`_ instances

        Returns
        -------
        set
            Representation of the hypergraph as a set of `gringo.Fun`_ instances


        .. _gringo.Fun: http://potassco.sourceforge.net/gringo.html#Fun
        """
        fs = set()
        for i, n in self.nodes.iteritems():
            fs.add(gringo.Fun('node', [n, i]))

        for j, i in self.hyper.iteritems():
            fs.add(gringo.Fun('hyper', [i, j, len(self.edges[self.edges.hyper_idx == j])]))

        for j, v, s in self.edges.itertuples(index=False):
            fs.add(gringo.Fun('edge', [j, v, s]))

        return fs
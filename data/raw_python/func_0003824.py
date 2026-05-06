def from_molecular_graph(cls, molecular_graph, labels=None):
        """Initialize a similarity descriptor

           Arguments:
             molecular_graphs  --  A MolecularGraphs object
             labels  --  a list with integer labels used to identify atoms of
                         the same type. When not given, the atom numbers from
                         the molecular graph are used.
        """
        if labels is None:
            labels = molecular_graph.numbers.astype(int)
        return cls(molecular_graph.distances, labels)
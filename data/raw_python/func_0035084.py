def to_graph_tool(self):
    '''Converts this Graph object to a graph_tool-compatible object.
    Requires the graph_tool library.
    Note that the internal ordering of graph_tool seems to be column-major.'''
    # Import here to avoid ImportErrors when graph_tool isn't available.
    import graph_tool
    gt = graph_tool.Graph(directed=self.is_directed())
    gt.add_edge_list(self.pairs())
    if self.is_weighted():
      weights = gt.new_edge_property('double')
      for e,w in zip(gt.edges(), self.edge_weights()):
        weights[e] = w
      gt.edge_properties['weight'] = weights
    return gt
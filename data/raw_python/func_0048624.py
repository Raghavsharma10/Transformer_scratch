def get_root_graph(self,root):
      """Return back a graph containing just the root and children"""
      children = self.get_children(root)
      g = Graph()
      nodes = [root]+children
      for node in nodes: g.add_node(node)
      node_ids = [x.id for x in nodes]
      edges = [x for x in self._edges.values() if x.node1.id in node_ids and x.node2.id in node_ids]
      for e in edges: g.add_edge(e)
      return g
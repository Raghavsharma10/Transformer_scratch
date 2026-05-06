def remove_node(self,node):
      """remove the node"""
      if node.id not in self._nodes: return      
      """find edges to remove"""
      edges = set()
      for e in self._edges.values():
         if e.node1.id == node.id: edges.add(e.id)
         if e.node2.id == node.id: edges.add(e.id)
      edges = [self._edges[x] for x in list(edges)]
      for e in edges: self.remove_edge(e)
      del self._nodes[node.id]
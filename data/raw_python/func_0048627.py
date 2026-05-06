def remove_edge(self,edge):
      """Remove the edge"""
      if edge.id not in self._edges: return # its not in the graph
      del self._p2c[edge.node1.id][edge.node2.id][edge.id]
      if len(self._p2c[edge.node1.id][edge.node2.id].keys()) == 0:
         del self._p2c[edge.node1.id][edge.node2.id]
      if len(self._p2c[edge.node1.id].keys()) == 0:
         del self._p2c[edge.node1.id]

      del self._c2p[edge.node2.id][edge.node1.id][edge.id]
      if len(self._c2p[edge.node2.id][edge.node1.id].keys()) == 0:
         del self._c2p[edge.node2.id][edge.node1.id]
      if len(self._c2p[edge.node2.id].keys()) == 0:
         del self._c2p[edge.node2.id]

      del self._edges[edge.id]
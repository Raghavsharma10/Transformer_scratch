def move_edges(self,n1,n2):
      """Move edges from node 1 to node 2
 
         Not self edges though

         Overwrites edges
      """
      #Traverse edges to find incoming with n1
      incoming = []
      for e in self._edges.values():
         if e.node2.id == n1.id: incoming.append(e)
      #Traverse edges to find outgoing from n1
      outgoing = []
      for e in self._edges.values():
         if e.node1.id == n1.id: outgoing.append(e)
      #Make new edges to the new target
      for e in incoming:
         if e.node1.id == n2.id: continue # skip self
         newedge = Edge(e.node1,n2,payload_list=n2.payload_list+n1.payload_list)
         self.add_edge(newedge)
      for e in outgoing:
         if e.node2.id == n2.id: continue # skip self
         newedge = Edge(n2,e.node2,payload_list=n2.payload_list+n1.payload_list)
         self.add_edge(newedge)
      #now remove the edges that got transfered
      for e in incoming: self.remove_edge(e)
      for e in outgoing: self.remove_edge(e)
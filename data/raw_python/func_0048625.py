def merge_cycles(self):
      """Work on this graph and remove cycles, with nodes containing concatonated lists of payloads"""
      while True:
         ### remove any self edges
         own_edges = self.get_self_edges()
         if len(own_edges) > 0:
            for e in own_edges: self.remove_edge(e)
         c = self.find_cycle()
         if not c: return
         keep = c[0]
         remove_list = c[1:]
         for n in remove_list: self.move_edges(n,keep)
         for n in remove_list: keep.payload_list += n.payload_list
         for n in remove_list: self.remove_node(n)
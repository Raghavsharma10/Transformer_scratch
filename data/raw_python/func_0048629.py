def find_cycle(self):
      """greedy search for a cycle"""
      for node in self.nodes:
         cyc = self._follow_children(node)
         if len(cyc) > 0:
            return [self._nodes[x] for x in cyc]
      return None
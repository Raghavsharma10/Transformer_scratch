def roots(self):
      """get the nodes with no children"""
      return [x for x in self._nodes.values() if x.id not in self._c2p]
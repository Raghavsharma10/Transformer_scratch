def enterTuple(self, tuple, path):
    """Called for every tuple.

    If this returns False, the elements of the tuple will not be recursed over
    and leaveTuple() will not be called.
    """
    if skip_name(path):
      return False
    node = Node(path, tuple)
    if self.condition.matches(node):
      self.unordered.append(node)
      return False
    return True
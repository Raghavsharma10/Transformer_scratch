def write_dag(self):
    """
    Write either a dag or a dax.
    """
    if not self.__nodes_finalized:
      for node in self.__nodes:
        node.finalize()
    self.write_concrete_dag()
    self.write_abstract_dag()
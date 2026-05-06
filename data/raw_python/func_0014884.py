def put(self, name_to_val):
    """ Loads a dictionary of variable names into the matlab com client.
    """
    self._check_open()
    for name, val in name_to_val.iteritems():
      # First try to put data as a matrix:
      try:
        self.client.PutFullMatrix(name, 'base', val, None)
      except:
        self.client.PutWorkspaceData(name, 'base', val)
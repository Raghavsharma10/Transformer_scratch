def _methodInTraceback(self, name, traceback):
    '''
    Returns boolean whether traceback contains method from this instance
    '''
    foundMethod = False
    for frame in self._frames(traceback):
      this = frame.f_locals.get('self')
      if this is self and frame.f_code.co_name == name:
        foundMethod = True
        break
    return foundMethod
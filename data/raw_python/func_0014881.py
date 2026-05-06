def open(self, visible=False):
    """ Dispatches the matlab COM client.

    Note: If this method fails, try running matlab with the -regserver flag.
    """
    if self.client:
      raise MatlabConnectionError('Matlab(TM) COM client is still active. Use close to '
                      'close it')
    self.client = win32com.client.Dispatch('matlab.application')
    self.client.visible = visible
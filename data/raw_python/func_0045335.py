def header(self):
    '''
    This returns the first header in the data file

    '''
    if self._header is None:
      self._header = self._read_half_frame_header(self.data)

    return self._header
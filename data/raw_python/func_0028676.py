def close_temp_file(self):
    """
    <Purpose>
      Closes the temporary file object. 'close_temp_file' mimics usual
      file.close(), however temporary file destroys itself when
      'close_temp_file' is called. Further if compression is set, second
      temporary file instance 'self._orig_file' is also closed so that no open
      temporary files are left open.

    <Arguments>
      None.

    <Exceptions>
      None.

    <Side Effects>
      Closes 'self._orig_file'.

    <Return>
      None.
    """

    self.temporary_file.close()
    # If compression has been set, we need to explicitly close the original
    # file object.
    if self._orig_file is not None:
      self._orig_file.close()
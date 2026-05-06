def write(self, data, auto_flush=True):
    """
    <Purpose>
      Writes a data string to the file.

    <Arguments>
      data:
        A string containing some data.

      auto_flush:
        Boolean argument, if set to 'True', all data will be flushed from
        internal buffer.

    <Exceptions>
      None.

    <Return>
      None.
    """

    self.temporary_file.write(data)
    if auto_flush:
      self.flush()
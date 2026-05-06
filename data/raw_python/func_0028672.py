def read(self, size=None):
    """
    <Purpose>
      Read specified number of bytes.  If size is not specified then the whole
      file is read and the file pointer is placed at the beginning of the file.

    <Arguments>
      size:
        Number of bytes to be read.

    <Exceptions>
      securesystemslib.exceptions.FormatError: if 'size' is invalid.

    <Return>
      String of data.
    """

    if size is None:
      self.temporary_file.seek(0)
      data = self.temporary_file.read()
      self.temporary_file.seek(0)

      return data

    else:
      if not (isinstance(size, int) and size > 0):
        raise securesystemslib.exceptions.FormatError

      return self.temporary_file.read(size)
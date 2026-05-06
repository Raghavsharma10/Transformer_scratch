def GetByteSize(self):
    """Retrieves the byte size of the data type definition.

    Returns:
      int: data type size in bytes or None if size cannot be determined.
    """
    if self.size == definitions.SIZE_NATIVE or self.units != 'bytes':
      return None

    return self.size
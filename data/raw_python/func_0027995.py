def GetByteSize(self):
    """Retrieves the byte size of the data type definition.

    Returns:
      int: data type size in bytes or None if size cannot be determined.
    """
    if self._byte_size is None and self.members:
      self._byte_size = 0
      for member_definition in self.members:
        byte_size = member_definition.GetByteSize()
        if byte_size is None:
          self._byte_size = None
          break

        self._byte_size = max(self._byte_size, byte_size)

    return self._byte_size
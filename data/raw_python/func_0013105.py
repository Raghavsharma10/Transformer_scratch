def reference(self):
    """Return the Reference object for this Key.

    This is a entity_pb.Reference instance -- a protocol buffer class
    used by the lower-level API to the datastore.

    NOTE: The caller should not mutate the return value.
    """
    if self.__reference is None:
      self.__reference = _ConstructReference(self.__class__,
                                             pairs=self.__pairs,
                                             app=self.__app,
                                             namespace=self.__namespace)
    return self.__reference
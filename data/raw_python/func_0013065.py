def _deserialize(self, entity, p, unused_depth=1):
    """Internal helper to deserialize this property from a protocol buffer.

    Subclasses may override this method.

    Args:
      entity: The entity, a Model (subclass) instance.
      p: A Property Message object (a protocol buffer).
      depth: Optional nesting depth, default 1 (unused here, but used
        by some subclasses that override this method).
    """
    if p.meaning() == entity_pb.Property.EMPTY_LIST:
      self._store_value(entity, [])
      return

    val = self._db_get_value(p.value(), p)
    if val is not None:
      val = _BaseValue(val)

    # TODO: replace the remainder of the function with the following commented
    # out code once its feasible to make breaking changes such as not calling
    # _store_value().

    # if self._repeated:
    #   entity._values.setdefault(self._name, []).append(val)
    # else:
    #   entity._values[self._name] = val

    if self._repeated:
      if self._has_value(entity):
        value = self._retrieve_value(entity)
        assert isinstance(value, list), repr(value)
        value.append(val)
      else:
        # We promote single values to lists if we are a list property
        value = [val]
    else:
      value = val
    self._store_value(entity, value)
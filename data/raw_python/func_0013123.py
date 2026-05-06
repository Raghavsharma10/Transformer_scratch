def _from_base_type(self, ent):
    """Convert a Model instance (entity) to a Message value."""
    if ent._projection:
      # Projection query result.  Reconstitute the message from the fields.
      return _projected_entity_to_message(ent, self._message_type)

    blob = ent.blob_
    if blob is not None:
      protocol = self._protocol_impl
    else:
      # Perhaps it was written using a different protocol.
      protocol = None
      for name in _protocols_registry.names:
        key = '__%s__' % name
        if key in ent._values:
          blob = ent._values[key]
          if isinstance(blob, model._BaseValue):
            blob = blob.b_val
          protocol = _protocols_registry.lookup_by_name(name)
          break
    if blob is None or protocol is None:
      return None  # This will reveal the underlying dummy model.
    msg = protocol.decode_message(self._message_type, blob)
    return msg
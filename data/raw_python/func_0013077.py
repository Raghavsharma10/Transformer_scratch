def _to_pb(self, pb=None, allow_partial=False, set_key=True):
    """Internal helper to turn an entity into an EntityProto protobuf."""
    if not allow_partial:
      self._check_initialized()
    if pb is None:
      pb = entity_pb.EntityProto()

    if set_key:
      # TODO: Move the key stuff into ModelAdapter.entity_to_pb()?
      self._key_to_pb(pb)

    for unused_name, prop in sorted(self._properties.iteritems()):
      prop._serialize(self, pb, projection=self._projection)

    return pb
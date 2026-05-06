def _from_pb(cls, pb, set_key=True, ent=None, key=None):
    """Internal helper to create an entity from an EntityProto protobuf."""
    if not isinstance(pb, entity_pb.EntityProto):
      raise TypeError('pb must be a EntityProto; received %r' % pb)
    if ent is None:
      ent = cls()

    # A key passed in overrides a key in the pb.
    if key is None and pb.key().path().element_size():
      key = Key(reference=pb.key())
    # If set_key is not set, skip a trivial incomplete key.
    if key is not None and (set_key or key.id() or key.parent()):
      ent._key = key

    # NOTE(darke): Keep a map from (indexed, property name) to the property.
    # This allows us to skip the (relatively) expensive call to
    # _get_property_for for repeated fields.
    _property_map = {}
    projection = []
    for indexed, plist in ((True, pb.property_list()),
                           (False, pb.raw_property_list())):
      for p in plist:
        if p.meaning() == entity_pb.Property.INDEX_VALUE:
          projection.append(p.name())
        property_map_key = (p.name(), indexed)
        if property_map_key not in _property_map:
          _property_map[property_map_key] = ent._get_property_for(p, indexed)
        _property_map[property_map_key]._deserialize(ent, p)

    ent._set_projection(projection)
    return ent
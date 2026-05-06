def _serialize(self, entity, pb, prefix='', parent_repeated=False,
                 projection=None):
    """Internal helper to serialize this property to a protocol buffer.

    Subclasses may override this method.

    Args:
      entity: The entity, a Model (subclass) instance.
      pb: The protocol buffer, an EntityProto instance.
      prefix: Optional name prefix used for StructuredProperty
        (if present, must end in '.').
      parent_repeated: True if the parent (or an earlier ancestor)
        is a repeated Property.
      projection: A list or tuple of strings representing the projection for
        the model instance, or None if the instance is not a projection.
    """
    values = self._get_base_value_unwrapped_as_list(entity)
    name = prefix + self._name
    if projection and name not in projection:
      return

    if self._indexed:
      create_prop = lambda: pb.add_property()
    else:
      create_prop = lambda: pb.add_raw_property()

    if self._repeated and not values and self._write_empty_list:
      # We want to write the empty list
      p = create_prop()
      p.set_name(name)
      p.set_multiple(False)
      p.set_meaning(entity_pb.Property.EMPTY_LIST)
      p.mutable_value()
    else:
      # We write a list, or a single property
      for val in values:
        p = create_prop()
        p.set_name(name)
        p.set_multiple(self._repeated or parent_repeated)
        v = p.mutable_value()
        if val is not None:
          self._db_set_value(v, p, val)
          if projection:
            # Projected properties have the INDEX_VALUE meaning and only contain
            # the original property's name and value.
            new_p = entity_pb.Property()
            new_p.set_name(p.name())
            new_p.set_meaning(entity_pb.Property.INDEX_VALUE)
            new_p.set_multiple(False)
            new_p.mutable_value().CopyFrom(v)
            p.CopyFrom(new_p)
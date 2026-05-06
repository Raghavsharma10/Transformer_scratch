def _key_to_pb(self, pb):
    """Internal helper to copy the key into a protobuf."""
    key = self._key
    if key is None:
      pairs = [(self._get_kind(), None)]
      ref = key_module._ReferenceFromPairs(pairs, reference=pb.mutable_key())
    else:
      ref = key.reference()
      pb.mutable_key().CopyFrom(ref)
    group = pb.mutable_entity_group()  # Must initialize this.
    # To work around an SDK issue, only set the entity group if the
    # full key is complete.  TODO: Remove the top test once fixed.
    if key is not None and key.id():
      elem = ref.path().element(0)
      if elem.id() or elem.name():
        group.add_element().CopyFrom(elem)
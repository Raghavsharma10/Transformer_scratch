def _from_pb(cls, pb, set_key=True, ent=None, key=None):
    """Override.

    Use the class map to give the entity the correct subclass.
    """
    prop_name = cls.class_._name
    class_name = []
    for plist in [pb.property_list(), pb.raw_property_list()]:
      for p in plist:
        if p.name() == prop_name:
          class_name.append(p.value().stringvalue())
    cls = cls._class_map.get(tuple(class_name), cls)
    return super(PolyModel, cls)._from_pb(pb, set_key, ent, key)
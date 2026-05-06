def _update_kind_map(cls):
    """Override; called by Model._fix_up_properties().

    Update the kind map as well as the class map, except for PolyModel
    itself (its class key is empty).  Note that the kind map will
    contain entries for all classes in a PolyModel hierarchy; they all
    have the same kind, but different class names.  PolyModel class
    names, like regular Model class names, must be globally unique.
    """
    cls._kind_map[cls._class_name()] = cls
    class_key = cls._class_key()
    if class_key:
      cls._class_map[tuple(class_key)] = cls
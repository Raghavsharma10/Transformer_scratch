def _get_kind(cls):
    """Override.

    Make sure that the kind returned is the root class of the
    polymorphic hierarchy.
    """
    bases = cls._get_hierarchy()
    if not bases:
      # We have to jump through some hoops to call the superclass'
      # _get_kind() method.  First, this is called by the metaclass
      # before the PolyModel name is defined, so it can't use
      # super(PolyModel, cls)._get_kind().  Second, we can't just call
      # Model._get_kind() because that always returns 'Model'.  Hence
      # the 'im_func' hack.
      return model.Model._get_kind.im_func(cls)
    else:
      return bases[0]._class_name()
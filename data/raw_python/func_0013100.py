def _parse_from_ref(cls, pairs=None, flat=None,
                      reference=None, serialized=None, urlsafe=None,
                      app=None, namespace=None, parent=None):
    """Construct a Reference; the signature is the same as for Key."""
    if cls is not Key:
      raise TypeError('Cannot construct Key reference on non-Key class; '
                      'received %r' % cls)
    if (bool(pairs) + bool(flat) + bool(reference) + bool(serialized) +
        bool(urlsafe) + bool(parent)) != 1:
      raise TypeError('Cannot construct Key reference from incompatible '
                      'keyword arguments.')
    if urlsafe:
      serialized = _DecodeUrlSafe(urlsafe)
    if serialized:
      reference = _ReferenceFromSerialized(serialized)
    if reference:
      reference = _ReferenceFromReference(reference)
    pairs = []
    elem = None
    path = reference.path()
    for elem in path.element_list():
      kind = elem.type()
      if elem.has_id():
        id_or_name = elem.id()
      else:
        id_or_name = elem.name()
      if not id_or_name:
        id_or_name = None
      tup = (kind, id_or_name)
      pairs.append(tup)
    if elem is None:
      raise RuntimeError('Key reference has no path or elements (%r, %r, %r).'
                         % (urlsafe, serialized, str(reference)))
    # TODO: ensure that each element has a type and either an id or a name
    # You needn't specify app= or namespace= together with reference=,
    # serialized= or urlsafe=, but if you do, their values must match
    # what is already in the reference.
    ref_app = reference.app()
    if app is not None:
      if app != ref_app:
        raise RuntimeError('Key reference constructed uses a different app %r '
                           'than the one specified %r' %
                           (ref_app, app))
    ref_namespace = reference.name_space()
    if namespace is not None:
      if namespace != ref_namespace:
        raise RuntimeError('Key reference constructed uses a different '
                           'namespace %r than the one specified %r' %
                           (ref_namespace, namespace))
    return (reference, tuple(pairs), ref_app, ref_namespace)
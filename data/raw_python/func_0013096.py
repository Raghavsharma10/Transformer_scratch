def _ConstructReference(cls, pairs=None, flat=None,
                        reference=None, serialized=None, urlsafe=None,
                        app=None, namespace=None, parent=None):
  """Construct a Reference; the signature is the same as for Key."""
  if cls is not Key:
    raise TypeError('Cannot construct Key reference on non-Key class; '
                    'received %r' % cls)
  if (bool(pairs) + bool(flat) + bool(reference) + bool(serialized) +
      bool(urlsafe)) != 1:
    raise TypeError('Cannot construct Key reference from incompatible keyword '
                    'arguments.')
  if flat or pairs:
    if flat:
      if len(flat) % 2:
        raise TypeError('_ConstructReference() must have an even number of '
                        'positional arguments.')
      pairs = [(flat[i], flat[i + 1]) for i in xrange(0, len(flat), 2)]
    elif parent is not None:
      pairs = list(pairs)
    if not pairs:
      raise TypeError('Key references must consist of at least one pair.')
    if parent is not None:
      if not isinstance(parent, Key):
        raise datastore_errors.BadValueError(
            'Expected Key instance, got %r' % parent)
      pairs[:0] = parent.pairs()
      if app:
        if app != parent.app():
          raise ValueError('Cannot specify a different app %r '
                           'than the parent app %r' %
                           (app, parent.app()))
      else:
        app = parent.app()
      if namespace is not None:
        if namespace != parent.namespace():
          raise ValueError('Cannot specify a different namespace %r '
                           'than the parent namespace %r' %
                           (namespace, parent.namespace()))
      else:
        namespace = parent.namespace()
    reference = _ReferenceFromPairs(pairs, app=app, namespace=namespace)
  else:
    if parent is not None:
      raise TypeError('Key reference cannot be constructed when the parent '
                      'argument is combined with either reference, serialized '
                      'or urlsafe arguments.')
    if urlsafe:
      serialized = _DecodeUrlSafe(urlsafe)
    if serialized:
      reference = _ReferenceFromSerialized(serialized)
    if not reference.path().element_size():
      raise RuntimeError('Key reference has no path or elements (%r, %r, %r).'
                         % (urlsafe, serialized, str(reference)))
    # TODO: ensure that each element has a type and either an id or a name
    if not serialized:
      reference = _ReferenceFromReference(reference)
    # You needn't specify app= or namespace= together with reference=,
    # serialized= or urlsafe=, but if you do, their values must match
    # what is already in the reference.
    if app is not None:
      ref_app = reference.app()
      if app != ref_app:
        raise RuntimeError('Key reference constructed uses a different app %r '
                           'than the one specified %r' %
                           (ref_app, app))
    if namespace is not None:
      ref_namespace = reference.name_space()
      if namespace != ref_namespace:
        raise RuntimeError('Key reference constructed uses a different '
                           'namespace %r than the one specified %r' %
                           (ref_namespace, namespace))
  return reference
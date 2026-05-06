def _ReferenceFromPairs(pairs, reference=None, app=None, namespace=None):
  """Construct a Reference from a list of pairs.

  If a Reference is passed in as the second argument, it is modified
  in place.  The app and namespace are set from the corresponding
  keyword arguments, with the customary defaults.
  """
  if reference is None:
    reference = entity_pb.Reference()
  path = reference.mutable_path()
  last = False
  for kind, idorname in pairs:
    if last:
      raise datastore_errors.BadArgumentError(
          'Incomplete Key entry must be last')
    t = type(kind)
    if t is str:
      pass
    elif t is unicode:
      kind = kind.encode('utf8')
    else:
      if issubclass(t, type):
        # Late import to avoid cycles.
        from .model import Model
        modelclass = kind
        if not issubclass(modelclass, Model):
          raise TypeError('Key kind must be either a string or subclass of '
                          'Model; received %r' % modelclass)
        kind = modelclass._get_kind()
        t = type(kind)
      if t is str:
        pass
      elif t is unicode:
        kind = kind.encode('utf8')
      elif issubclass(t, str):
        pass
      elif issubclass(t, unicode):
        kind = kind.encode('utf8')
      else:
        raise TypeError('Key kind must be either a string or subclass of Model;'
                        ' received %r' % kind)
    # pylint: disable=superfluous-parens
    if not (1 <= len(kind) <= _MAX_KEYPART_BYTES):
      raise ValueError('Key kind string must be a non-empty string up to %i'
                       'bytes; received %s' %
                       (_MAX_KEYPART_BYTES, kind))
    elem = path.add_element()
    elem.set_type(kind)
    t = type(idorname)
    if t is int or t is long:
      # pylint: disable=superfluous-parens
      if not (1 <= idorname < _MAX_LONG):
        raise ValueError('Key id number is too long; received %i' % idorname)
      elem.set_id(idorname)
    elif t is str:
      # pylint: disable=superfluous-parens
      if not (1 <= len(idorname) <= _MAX_KEYPART_BYTES):
        raise ValueError('Key name strings must be non-empty strings up to %i '
                         'bytes; received %s' %
                         (_MAX_KEYPART_BYTES, idorname))
      elem.set_name(idorname)
    elif t is unicode:
      idorname = idorname.encode('utf8')
      # pylint: disable=superfluous-parens
      if not (1 <= len(idorname) <= _MAX_KEYPART_BYTES):
        raise ValueError('Key name unicode strings must be non-empty strings up'
                         ' to %i bytes; received %s' %
                         (_MAX_KEYPART_BYTES, idorname))
      elem.set_name(idorname)
    elif idorname is None:
      last = True
    elif issubclass(t, (int, long)):
      # pylint: disable=superfluous-parens
      if not (1 <= idorname < _MAX_LONG):
        raise ValueError('Key id number is too long; received %i' % idorname)
      elem.set_id(idorname)
    elif issubclass(t, basestring):
      if issubclass(t, unicode):
        idorname = idorname.encode('utf8')
      # pylint: disable=superfluous-parens
      if not (1 <= len(idorname) <= _MAX_KEYPART_BYTES):
        raise ValueError('Key name strings must be non-empty strings up to %i '
                         'bytes; received %s' % (_MAX_KEYPART_BYTES, idorname))
      elem.set_name(idorname)
    else:
      raise TypeError('id must be either a numeric id or a string name; '
                      'received %r' % idorname)
  # An empty app id means to use the default app id.
  if not app:
    app = _DefaultAppId()
  # Always set the app id, since it is mandatory.
  reference.set_app(app)
  # An empty namespace overrides the default namespace.
  if namespace is None:
    namespace = _DefaultNamespace()
  # Only set the namespace if it is not empty.
  if namespace:
    reference.set_name_space(namespace)
  return reference
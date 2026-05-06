def pick(source, *keys, **kws):
  '''
  Given a `source` dict or object, returns a new dict that contains a
  subset of keys (each key is a separate positional argument) and/or
  where each key is a string and has the specified `prefix`, specified
  as a keyword argument. Also accepts the optional keyword argument
  `dict` which must be a dict-like class that will be used to
  instantiate the returned object. Note that if `source` is an object
  without an `items()` iterator, then the selected keys will be
  extracted as attributes. The `prefix` keyword only works with
  dict-like objects. If the `tree` keyword is specified and set to
  truthy, each key is evaluated as a hierchical key walker spec. In
  other words, the following are equivalent:

  .. code:: python

    src = dict(a=dict(b='bee', c='cee'), d='dee')
    assert morph.pick(src, 'a.b', tree=True) == dict(a=dict(b='bee'))

  Requests for keys not found in `source` are silently ignored.

  :Changes:

  * `tree` support added in version 0.1.3.
  '''
  rettype = kws.pop('dict', dict)
  prefix  = kws.pop('prefix', None)
  tree    = kws.pop('tree', False)
  if kws:
    raise ValueError('invalid pick keyword arguments: %r' % (kws.keys(),))
  if prefix is not None and tree:
    raise ValueError('`prefix` and `tree` currently cannot be used together')
  if not source:
    return rettype()
  if prefix is not None:
    try:
      items = source.items()
    except AttributeError:
      items = None
    if items is not None:
      source = {k[len(prefix):]: v
                for k, v in items
                if getattr(k, 'startswith', lambda x: False)(prefix)}
    else:
      source = {attr[len(prefix):]: getattr(source, attr)
                for attr in properties(source)
                if attr.startswith(prefix)}
  if len(keys) <= 0:
    if prefix is not None:
      return rettype(source)
    return rettype()
  rkeys = keys
  if tree:
    rkeys = [key.split('.', 1)[0] for key in rkeys]
  try:
    ret = rettype({k: v for k, v in source.items() if k in rkeys})
  except AttributeError:
    ret = rettype({k: getattr(source, k) for k in rkeys if hasattr(source, k)})
  if tree:
    for key in keys:
      if '.' in key:
        key, rem = key.split('.', 1)
        if key in ret:
          ret[key] = pick(ret[key], rem, dict=rettype, prefix=prefix, tree=tree)
  return ret
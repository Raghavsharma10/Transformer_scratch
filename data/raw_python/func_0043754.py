def unflatten(obj):
  '''
  TODO: add docs
  '''
  if not isdict(obj):
    raise ValueError(
      'only dict-like objects can be unflattened, not %r' % (obj,))
  ret = dict()
  sub = dict()
  for key, value in obj.items():
    if '.' not in key and '[' not in key:
      ret[key] = value
      continue
    if '.' in key and '[' in key:
      idx = min(key.find('.'), key.find('['))
    elif '.' in key:
      idx = key.find('.')
    else:
      idx = key.find('[')
    prefix = key[:idx]
    if prefix not in sub:
      sub[prefix] = dict()
    sub[prefix][key[idx:]] = value
  for pfx, values in sub.items():
    if pfx in ret:
      raise ValueError(
        'conflicting scalar vs. structure for prefix: %s' % (pfx,))
    ret[pfx] = _relunflatten(pfx, values)
  return ret
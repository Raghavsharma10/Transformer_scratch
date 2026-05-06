def flatten(obj):
  '''
  TODO: add docs
  '''
  if isseq(obj):
    ret = []
    for item in obj:
      if isseq(item):
        ret.extend(flatten(item))
      else:
        ret.append(item)
    return ret
  if isdict(obj):
    ret = dict()
    for key, value in obj.items():
      for skey, sval in _relflatten(value):
        ret[key + skey] = sval
    return ret
  raise ValueError(
    'only list- and dict-like objects can be flattened, not %r' % (obj,))
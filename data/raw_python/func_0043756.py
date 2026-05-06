def xform(value, xformer):
  '''
  Recursively transforms `value` by calling `xformer` on all
  keys & values in dictionaries and all values in sequences. Note
  that `xformer` will be passed each value to transform as the
  first parameter and other keyword parameters based on type. All
  transformers MUST support arbitrary additional parameters to stay
  future-proof.

  For sequences, `xformer` will be provided the following additional
  keyword parameters:

  * `index`: the index of the current value in the current sequence.
  * `seq`: the current sequence being transformed.
  * `root`: a reference to the original `value` passed to `xform`.

  For dictionaries, `xformer` will be provided the following
  additional keyword parameters:

  * `item_key`: ONLY provided if the value being transformed is a
    *value* in key-value dictionary pair.
  * `item_value`: ONLY provided if the value being transformed is a
    *key* in key-value dictionary pair.
  * `dict`: the current dictionary being transformed.
  * `root`: a reference to the original `value` passed to `xform`.

  Added in version 0.1.3.
  '''
  def _xform(curval, **kws):
    if isseq(curval):
      return [
        _xform(val, index=idx, seq=curval)
        for idx, val in enumerate(curval) ]
    if isdict(curval):
      return {
        _xform(key, item_value=val, dict=curval) : _xform(val, item_key=key, dict=curval)
        for key, val in curval.items() }
    return xformer(curval, root=value, **kws)
  return _xform(value)
def get_multi_async(cls, blob_keys, **ctx_options):
    """Async version of get_multi()."""
    for blob_key in blob_keys:
      if not isinstance(blob_key, (BlobKey, basestring)):
        raise TypeError('Expected blob key, got %r' % (blob_key,))
    if 'parent' in ctx_options:
      raise TypeError('Parent is not supported')
    blob_key_strs = map(str, blob_keys)
    keys = [model.Key(BLOB_INFO_KIND, id) for id in blob_key_strs]
    return model.get_multi_async(keys, **ctx_options)
def get_async(cls, blob_key, **ctx_options):
    """Async version of get()."""
    if not isinstance(blob_key, (BlobKey, basestring)):
      raise TypeError('Expected blob key, got %r' % (blob_key,))
    if 'parent' in ctx_options:
      raise TypeError('Parent is not supported')
    return cls.get_by_id_async(str(blob_key), **ctx_options)
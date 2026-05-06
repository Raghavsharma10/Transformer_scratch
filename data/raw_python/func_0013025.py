def delete_multi_async(blob_keys, **options):
  """Async version of delete_multi()."""
  if isinstance(blob_keys, (basestring, BlobKey)):
    raise TypeError('Expected a list, got %r' % (blob_key,))
  rpc = blobstore.create_rpc(**options)
  yield blobstore.delete_async(blob_keys, rpc=rpc)
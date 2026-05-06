def delete_async(blob_key, **options):
  """Async version of delete()."""
  if not isinstance(blob_key, (basestring, BlobKey)):
    raise TypeError('Expected blob key, got %r' % (blob_key,))
  rpc = blobstore.create_rpc(**options)
  yield blobstore.delete_async(blob_key, rpc=rpc)
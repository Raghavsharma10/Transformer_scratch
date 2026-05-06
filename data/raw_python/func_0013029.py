def fetch_data_async(blob, start_index, end_index, **options):
  """Async version of fetch_data()."""
  if isinstance(blob, BlobInfo):
    blob = blob.key()
  rpc = blobstore.create_rpc(**options)
  rpc = blobstore.fetch_data_async(blob, start_index, end_index, rpc=rpc)
  result = yield rpc
  raise tasklets.Return(result)
def create_upload_url_async(success_path,
                            max_bytes_per_blob=None,
                            max_bytes_total=None,
                            **options):
  """Async version of create_upload_url()."""
  rpc = blobstore.create_rpc(**options)
  rpc = blobstore.create_upload_url_async(success_path,
                                          max_bytes_per_blob=max_bytes_per_blob,
                                          max_bytes_total=max_bytes_total,
                                          rpc=rpc)
  result = yield rpc
  raise tasklets.Return(result)
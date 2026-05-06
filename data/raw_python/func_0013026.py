def create_upload_url(success_path,
                      max_bytes_per_blob=None,
                      max_bytes_total=None,
                      **options):
  """Create upload URL for POST form.

  Args:
    success_path: Path within application to call when POST is successful
      and upload is complete.
    max_bytes_per_blob: The maximum size in bytes that any one blob in the
      upload can be or None for no maximum size.
    max_bytes_total: The maximum size in bytes that the aggregate sizes of all
      of the blobs in the upload can be or None for no maximum size.
    **options: Options for create_rpc().

  Returns:
    The upload URL.

  Raises:
    TypeError: If max_bytes_per_blob or max_bytes_total are not integral types.
    ValueError: If max_bytes_per_blob or max_bytes_total are not
      positive values.
  """
  fut = create_upload_url_async(success_path,
                                max_bytes_per_blob=max_bytes_per_blob,
                                max_bytes_total=max_bytes_total,
                                **options)
  return fut.get_result()
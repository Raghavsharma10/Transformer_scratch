def fetch_data(blob, start_index, end_index, **options):
  """Fetch data for blob.

  Fetches a fragment of a blob up to MAX_BLOB_FETCH_SIZE in length.  Attempting
  to fetch a fragment that extends beyond the boundaries of the blob will return
  the amount of data from start_index until the end of the blob, which will be
  a smaller size than requested.  Requesting a fragment which is entirely
  outside the boundaries of the blob will return empty string.  Attempting
  to fetch a negative index will raise an exception.

  Args:
    blob: BlobInfo, BlobKey, str or unicode representation of BlobKey of
      blob to fetch data from.
    start_index: Start index of blob data to fetch.  May not be negative.
    end_index: End index (inclusive) of blob data to fetch.  Must be
      >= start_index.
    **options: Options for create_rpc().

  Returns:
    str containing partial data of blob.  If the indexes are legal but outside
    the boundaries of the blob, will return empty string.

  Raises:
    TypeError if start_index or end_index are not indexes.  Also when blob
      is not a string, BlobKey or BlobInfo.
    DataIndexOutOfRangeError when start_index < 0 or end_index < start_index.
    BlobFetchSizeTooLargeError when request blob fragment is larger than
      MAX_BLOB_FETCH_SIZE.
    BlobNotFoundError when blob does not exist.
  """
  fut = fetch_data_async(blob, start_index, end_index, **options)
  return fut.get_result()
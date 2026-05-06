def _scratch_stream_name(self):
    """
    A unique cache stream name for this QueryCache.

    Hashes the necessary facts about this QueryCache to generate a
    unique cache stream name.  Different `query_function`
    implementations at different `bucket_width` values will be cached
    to different streams.

    TODO(marcua): This approach won't work for dynamically-generated
    functions.  We will want to either:
      1) Hash the function closure/containing scope.
      2) Ditch this approach and rely on the caller to tell us all the
         information that makes this function unique.
    """
    query_details = [
      str(QueryCache.QUERY_CACHE_VERSION),
      str(self._bucket_width),
      binascii.b2a_hex(marshal.dumps(self._query_function.func_code)),
      str(self._query_function_args),
      str(self._query_function_kwargs),
    ]
    return hashlib.sha512('$'.join(query_details)).hexdigest()[:20]
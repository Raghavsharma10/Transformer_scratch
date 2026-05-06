def get_streams(self, namespace=None):
    """
    Queries the Kronos server and fetches a list of streams available to be
    read.
    """
    request_dict = {}
    namespace = namespace or self.namespace
    if namespace is not None:
      request_dict['namespace'] = namespace
    response = self._make_request(self._streams_url,
                                  data=request_dict,
                                  stream=True)
    for line in response.iter_lines():
      if line:
        yield line
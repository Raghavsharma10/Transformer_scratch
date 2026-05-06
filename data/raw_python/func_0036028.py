def infer_schema(self, stream, namespace=None):
    """
    Queries the Kronos server and fetches the inferred schema for the
    requested stream.
    """
    return self._make_request(self._infer_schema_url,
                              data={'stream': stream,
                                    'namespace': namespace or self.namespace})
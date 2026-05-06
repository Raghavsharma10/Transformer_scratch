def infer_schema(environment, start_response, headers):
  """
  Return the inferred schema of the requested stream.
  POST body should contain a JSON encoded version of:
    { stream: stream_name,
      namespace: namespace_name (optional)
    }
  """
  stream = environment['json']['stream']
  namespace = environment['json'].get('namespace') or settings.default_namespace

  start_response('200 OK', headers)
  schema = _infer_schema(namespace, stream)
  response = {
    'stream': stream,
    'namespace': namespace,
    'schema': schema,
    SUCCESS_FIELD: True
  }
  return response
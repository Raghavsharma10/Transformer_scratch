def get_streams(environment, start_response, headers):
  """
  List all streams that can be read from Kronos right now.
  POST body should contain a JSON encoded version of:
    { namespace: namespace_name (optional)
    }
  """
  start_response('200 OK', headers)
  streams_seen_so_far = set()
  namespace = environment['json'].get('namespace', settings.default_namespace)
  for prefix, backend in router.get_read_backends(namespace):
    for stream in backend.streams(namespace):
      if stream.startswith(prefix) and stream not in streams_seen_so_far:
        streams_seen_so_far.add(stream)
        yield '{0}\r\n'.format(stream)
  yield ''
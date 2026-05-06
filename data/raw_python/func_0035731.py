def index(environment, start_response, headers):
  """
  Return the status of this Kronos instance + its backends>
  Doesn't expect any URL parameters.
  """
  response = {'service': 'kronosd',
              'version': kronos.__version__,
              'id': settings.node['id'],
              'storage': {},
              SUCCESS_FIELD: True}

  # Check if each backend is alive
  for name, backend in router.get_backends():
    response['storage'][name] = {'alive': backend.is_alive(),
                                 'backend': settings.storage[name]['backend']}

  start_response('200 OK', headers)
  return response
def make_connection(config=None, default_model=None,
                    _api_version=datastore_rpc._DATASTORE_V3,
                    _id_resolver=None):
  """Create a new Connection object with the right adapter.

  Optionally you can pass in a datastore_rpc.Configuration object.
  """
  return datastore_rpc.Connection(
      adapter=ModelAdapter(default_model, id_resolver=_id_resolver),
      config=config,
      _api_version=_api_version)
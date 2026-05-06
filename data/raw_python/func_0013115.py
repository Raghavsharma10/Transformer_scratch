def _make_cloud_datastore_context(app_id, external_app_ids=()):
  """Creates a new context to connect to a remote Cloud Datastore instance.

  This should only be used outside of Google App Engine.

  Args:
    app_id: The application id to connect to. This differs from the project
      id as it may have an additional prefix, e.g. "s~" or "e~".
    external_app_ids: A list of apps that may be referenced by data in your
      application. For example, if you are connected to s~my-app and store keys
      for s~my-other-app, you should include s~my-other-app in the external_apps
      list.
  Returns:
    An ndb.Context that can connect to a Remote Cloud Datastore. You can use
    this context by passing it to ndb.set_context.
  """
  from . import model  # Late import to deal with circular imports.
  # Late import since it might not exist.
  if not datastore_pbs._CLOUD_DATASTORE_ENABLED:
    raise datastore_errors.BadArgumentError(
        datastore_pbs.MISSING_CLOUD_DATASTORE_MESSAGE)
  import googledatastore
  try:
    from google.appengine.datastore import cloud_datastore_v1_remote_stub
  except ImportError:
    from google3.apphosting.datastore import cloud_datastore_v1_remote_stub

  current_app_id = os.environ.get('APPLICATION_ID', None)
  if current_app_id and current_app_id != app_id:
    # TODO(pcostello): We should support this so users can connect to different
    # applications.
    raise ValueError('Cannot create a Cloud Datastore context that connects '
                     'to an application (%s) that differs from the application '
                     'already connected to (%s).' % (app_id, current_app_id))
  os.environ['APPLICATION_ID'] = app_id

  id_resolver = datastore_pbs.IdResolver((app_id,) + tuple(external_app_ids))
  project_id = id_resolver.resolve_project_id(app_id)
  endpoint = googledatastore.helper.get_project_endpoint_from_env(project_id)
  datastore = googledatastore.Datastore(
      project_endpoint=endpoint,
      credentials=googledatastore.helper.get_credentials_from_env())

  conn = model.make_connection(_api_version=datastore_rpc._CLOUD_DATASTORE_V1,
                               _id_resolver=id_resolver)

  # If necessary, install the stubs
  try:
    stub = cloud_datastore_v1_remote_stub.CloudDatastoreV1RemoteStub(datastore)
    apiproxy_stub_map.apiproxy.RegisterStub(datastore_rpc._CLOUD_DATASTORE_V1,
                                            stub)
  except:
    pass  # The stub is already installed.
  # TODO(pcostello): Ensure the current stub is connected to the right project.

  # Install a memcache and taskqueue stub which throws on everything.
  try:
    apiproxy_stub_map.apiproxy.RegisterStub('memcache', _ThrowingStub())
  except:
    pass  # The stub is already installed.
  try:
    apiproxy_stub_map.apiproxy.RegisterStub('taskqueue', _ThrowingStub())
  except:
    pass  # The stub is already installed.


  return make_context(conn=conn)
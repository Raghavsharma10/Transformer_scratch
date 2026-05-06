def synctasklet(func):
  """Decorator to run a function as a tasklet when called.

  Use this to wrap a request handler function that will be called by
  some web application framework (e.g. a Django view function or a
  webapp.RequestHandler.get method).
  """
  taskletfunc = tasklet(func)  # wrap at declaration time.

  @utils.wrapping(func)
  def synctasklet_wrapper(*args, **kwds):
    # pylint: disable=invalid-name
    __ndb_debug__ = utils.func_info(func)
    return taskletfunc(*args, **kwds).get_result()
  return synctasklet_wrapper
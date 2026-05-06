def toplevel(func):
  """A sync tasklet that sets a fresh default Context.

  Use this for toplevel view functions such as
  webapp.RequestHandler.get() or Django view functions.
  """
  synctaskletfunc = synctasklet(func)  # wrap at declaration time.

  @utils.wrapping(func)
  def add_context_wrapper(*args, **kwds):
    # pylint: disable=invalid-name
    __ndb_debug__ = utils.func_info(func)
    _state.clear_all_pending()
    # Create and install a new context.
    ctx = make_default_context()
    try:
      set_context(ctx)
      return synctaskletfunc(*args, **kwds)
    finally:
      set_context(None)
      ctx.flush().check_success()
      eventloop.run()  # Ensure writes are flushed, etc.
  return add_context_wrapper
def _transfer_result(fut1, fut2):
  """Helper to transfer result or errors from one Future to another."""
  exc = fut1.get_exception()
  if exc is not None:
    tb = fut1.get_traceback()
    fut2.set_exception(exc, tb)
  else:
    val = fut1.get_result()
    fut2.set_result(val)
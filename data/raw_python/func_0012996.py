def _finished_callback(self, batch_fut, todo):
    """Passes exception along.

    Args:
      batch_fut: the batch future returned by running todo_tasklet.
      todo: (fut, option) pair. fut is the future return by each add() call.

    If the batch fut was successful, it has already called fut.set_result()
    on other individual futs. This method only handles when the batch fut
    encountered an exception.
    """
    self._running.remove(batch_fut)
    err = batch_fut.get_exception()
    if err is not None:
      tb = batch_fut.get_traceback()
      for (fut, _) in todo:
        if not fut.done():
          fut.set_exception(err, tb)
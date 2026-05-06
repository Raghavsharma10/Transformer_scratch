def add(self, arg, options=None):
    """Adds an arg and gets back a future.

    Args:
      arg: one argument for _todo_tasklet.
      options: rpc options.

    Return:
      An instance of future, representing the result of running
        _todo_tasklet without batching.
    """
    fut = tasklets.Future('%s.add(%s, %s)' % (self, arg, options))
    todo = self._queues.get(options)
    if todo is None:
      utils.logging_debug('AutoBatcher(%s): creating new queue for %r',
                          self._todo_tasklet.__name__, options)
      if not self._queues:
        eventloop.add_idle(self._on_idle)
      todo = self._queues[options] = []
    todo.append((fut, arg))
    if len(todo) >= self._limit:
      del self._queues[options]
      self.run_queue(options, todo)
    return fut
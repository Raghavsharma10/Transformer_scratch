def transactional_tasklet(func, args, kwds, **options):
  """The async version of @ndb.transaction.

  Will return the result of the wrapped function as a Future.
  """
  from . import tasklets
  func = tasklets.tasklet(func)
  return transactional_async.wrapped_decorator(func, args, kwds, **options)
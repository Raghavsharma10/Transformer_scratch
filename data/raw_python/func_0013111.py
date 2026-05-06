def sleep(dt):
  """Public function to sleep some time.

  Example:
    yield tasklets.sleep(0.5)  # Sleep for half a sec.
  """
  fut = Future('sleep(%.3f)' % dt)
  eventloop.queue_call(dt, fut.set_result, None)
  return fut
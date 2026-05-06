def insort_event_right(self, event, lo=0, hi=None):
    """Insert event in queue, and keep it sorted assuming queue is sorted.

    If event is already in queue, insert it to the right of the rightmost
    event (to keep FIFO order).

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    Args:
      event: a (time in sec since unix epoch, callback, args, kwds) tuple.
    """

    if lo < 0:
      raise ValueError('lo must be non-negative')
    if hi is None:
      hi = len(self.queue)
    while lo < hi:
      mid = (lo + hi) // 2
      if event[0] < self.queue[mid][0]:
        hi = mid
      else:
        lo = mid + 1
    self.queue.insert(lo, event)
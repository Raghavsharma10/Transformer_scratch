def memoizing_fibonacci(n):
  """A memoizing recursive Fibonacci to exercise RPCs."""
  if n <= 1:
    raise ndb.Return(n)
  key = ndb.Key(FibonacciMemo, str(n))
  memo = yield key.get_async(ndb_should_cache=False)
  if memo is not None:
    assert memo.arg == n
    logging.info('memo hit: %d -> %d', n, memo.value)
    raise ndb.Return(memo.value)
  logging.info('memo fail: %d', n)
  a = yield memoizing_fibonacci(n - 1)
  b = yield memoizing_fibonacci(n - 2)
  ans = a + b
  memo = FibonacciMemo(key=key, arg=n, value=ans)
  logging.info('memo write: %d -> %d', n, memo.value)
  yield memo.put_async(ndb_should_cache=False)
  raise ndb.Return(ans)
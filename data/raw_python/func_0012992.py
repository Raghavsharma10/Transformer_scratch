def fibonacci(n):
  """A recursive Fibonacci to exercise task switching."""
  if n <= 1:
    raise ndb.Return(n)
  a, b = yield fibonacci(n - 1), fibonacci(n - 2)
  raise ndb.Return(a + b)
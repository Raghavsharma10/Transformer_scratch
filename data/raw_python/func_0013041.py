def transactional(func, args, kwds, **options):
  """Decorator to make a function automatically run in a transaction.

  Args:
    **ctx_options: Transaction options (see transaction(), but propagation
      default to TransactionOptions.ALLOWED).

  This supports two forms:

  (1) Vanilla:
      @transactional
      def callback(arg):
        ...

  (2) With options:
      @transactional(retries=1)
      def callback(arg):
        ...
  """
  return transactional_async.wrapped_decorator(
      func, args, kwds, **options).get_result()
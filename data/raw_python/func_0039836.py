def retryable(retryer=retry_ex, times=3, cap=120000):
    """
    A decorator to make a function retry. By default the retry
    occurs when an exception is thrown, but this may be changed
    by modifying the ``retryer`` argument.

    See also :py:func:`retry_ex` and :py:func:`retry_bool`. By
    default :py:func:`retry_ex` is used as the retry function.

    Note that the decorator must be called even if not given
    keyword arguments.

    :param function retryer: A function to handle retries
    :param int times: Number of times to retry on initial failure
    :param int cap: Maximum wait time in milliseconds

    :Example:

    ::

      @retryable()
      def can_fail():
          ....

      @retryable(retryer=retry_bool, times=10)
      def can_fail_bool():
          ....
    """
    def _retryable(func):
        @f.wraps(func)
        def wrapper(*args, **kwargs):
            return retryer(lambda: func(*args, **kwargs), times, cap)
        return wrapper
    return _retryable
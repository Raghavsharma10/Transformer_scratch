def _check_stream_timeout(started, timeout):
    """Check if the timeout has been reached and raise a `StopIteration` if so.
    """
    if timeout:
        elapsed = datetime.datetime.utcnow() - started
        if elapsed.seconds > timeout:
            raise StopIteration
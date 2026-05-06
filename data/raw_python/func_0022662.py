def auto_retry(fun):
    """Decorator for retrying method calls, based on instance parameters."""

    @functools.wraps(fun)
    def decorated(instance, *args, **kwargs):
        """Wrapper around a decorated function."""
        cfg = instance._retry_config
        remaining_tries = cfg.retry_attempts
        current_wait = cfg.retry_wait
        retry_backoff = cfg.retry_backoff
        last_error = None

        while remaining_tries >= 0:
            try:
                return fun(instance, *args, **kwargs)
            except socket.error as e:
                last_error = e
                instance._retry_logger.warning('Connection failed: %s', e)

            remaining_tries -= 1
            if remaining_tries == 0:
                # Last attempt
                break

            # Wait a bit
            time.sleep(current_wait)
            current_wait *= retry_backoff

        # All attempts failed, let's raise the last error.
        raise last_error

    return decorated
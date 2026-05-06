def transaction_atomic_with_retry(num_retries=5, backoff=0.1):
    """
    This is a decorator that will wrap the decorated method in an atomic transaction and
    retry the transaction a given number of times

    :param num_retries: How many times should we retry before we give up
    :param backoff: How long should we wait after each try
    """

    # Create the decorator
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        # Keep track of how many times we have tried
        num_tries = 0
        exception = None

        # Call the main sync entities method and catch any exceptions
        while num_tries <= num_retries:
            # Try running the transaction
            try:
                with transaction.atomic():
                    return wrapped(*args, **kwargs)
            # Catch any operation errors
            except db.utils.OperationalError as e:
                num_tries += 1
                exception = e
                sleep(backoff * num_tries)

        # If we have an exception raise it
        raise exception

    # Return the decorator
    return wrapper
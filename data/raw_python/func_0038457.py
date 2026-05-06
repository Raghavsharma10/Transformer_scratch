def as_completed(fs, timeout=None):
    """An iterator over the given futures that yields each as it completes.

    Args:
        fs: The sequence of Futures (possibly created by different Executors) to
            iterate over.
        timeout: The maximum number of seconds to wait. If None, then there
            is no limit on the wait time.

    Returns:
        An iterator that yields the given Futures as they complete (finished or
        cancelled).

    Raises:
        TimeoutError: If the entire result iterator could not be generated
            before the given timeout.
    """
    with _AcquireFutures(fs):
        finished = set(f for f in fs if f._state in [CANCELLED_AND_NOTIFIED, FINISHED])
        pending = set(fs) - finished
        waiter = _create_and_install_waiters(fs, _AS_COMPLETED)

    timer = Timeout(timeout)
    timer.start()
    try:
        for future in finished:
            yield future
        while pending:
            waiter.event.wait()
            with waiter.lock:
                finished = waiter.finished_futures
                waiter.finished_futures = []
                waiter.event.clear()
            for future in finished:
                yield future
                pending.remove(future)
    except Timeout as e:
        if timer is not e:
            raise
        raise TimeoutError('%d (of %d) futures unfinished' % (len(pending), len(fs)))
    finally:
        timer.cancel()
        for f in fs:
            f._waiters.remove(waiter)